"""Canonical Python DSL printer for HIR Functions.

Converts a ``hir.Function`` to executable Python source using the
``@func`` DSL. Placement sugar names only explicit mesh-scope bindings;
otherwise the verbose ``ShardLayout(...)`` form is used.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from tilefoundry import services, utils
from tilefoundry.interfaces import valuation
from tilefoundry.ir import clause, core, expr, types
from tilefoundry.ir import utils as ir_utils
from tilefoundry.ir.expr import op_registry
from tilefoundry.ir.hir import tensor

from .module_names import definition_names, module_definition_order, slot_binding
from .print_context import HirPrintContext
from .printer_base import PARTS, PythonPrinter
from .tir_printer import function_block, tir_function_to_python, tir_module_to_python


class HirPrinter(PythonPrinter):
    """Canonical HIR expression, type, and function printer."""

    def __init__(self) -> None:
        super().__init__()
        self.names: dict[int, str] = {}
        self.definitions: dict[int, HirPrinter] = {}
        self.function: core.Function | None = None
        self.statements: dict[int, PrintedStatement] = {}
        self.param_alias: dict[int, expr.Expr] = {}
        self.child_entries: dict[int, str] = {}
        self.moved_window = lambda start, size, stride: None

    def print(self, value, ctx=None, indent: str = "", *, options=None) -> str:
        """A whole HIR function under *options*; any other value as ``PythonPrinter`` does."""
        if isinstance(value, core.SSAValue):
            text = self.print(value.expr, ctx, indent)
            owner = self.definitions.get(id(value.expr))
            if owner is not None and (statement := owner.statements.get(id(value.expr))) is not None:
                if tuple(owner.names.values()).count(owner.names[id(value.expr)]) > 1:
                    text += f":{statement.line}"
            return text if len(ir_utils.leaves_of(value.expr.type)) == 1 else f"{text}[{value.index}]"
        if isinstance(value, core.Function):
            return self.render(value, options=options).source
        if isinstance(value, expr.Expr) and id(value) in self.definitions:
            owner = self.definitions[id(value)]
            text = owner.reference(value, ctx)
            return text if owner.function is self.function else f"{owner.function.name}::{text}"
        return super().print(value, ctx, indent)

    def render(self, fn: core.Function, *, options=None, target=None) -> PythonRendering:
        if options is not None and options.comment_metadata_types:
            render_hir_function(fn, printer=self, target=target)
        return render_hir_function(fn, options=options, printer=self, target=target)

    def bind_def(
        self,
        names: dict[int, str],
        param_alias: dict[int, expr.Expr],
        child_entries: dict[int, str],
        moved_window,
    ) -> None:
        """Install the value-naming environment for one function definition."""
        self.names = names
        self.param_alias = param_alias
        self.child_entries = child_entries
        self.moved_window = moved_window

    def define(self, printer: HirPrinter, fn: core.Function, names: dict[int, str]) -> None:
        """Record *printer* as the owner of *fn* and of every value named in *names*."""
        self.definitions[id(fn)] = printer
        self.definitions.update({identity: printer for identity in names})

    def share(self, fn: core.Function, shared: HirPrinter) -> None:
        """Print *fn* while reading and recording definitions through *shared*."""
        self.function = fn
        self.definitions = shared.definitions
        self.statements = shared.statements

    def definition_printer(self, node) -> HirPrinter:
        """The printer that rendered the definition holding *node*."""
        return self.definitions[id(node)]

    def add_statements(self, statements: dict[int, PrintedStatement]) -> None:
        """Record the Call statements a rendering located."""
        self.statements.update(statements)

    def tuple_reference(self, elements, ctx=None) -> str:
        inner = ", ".join(
            self.print(element.value, ctx) if isinstance(element, expr.Constant) else self.reference(element, ctx)
            for element in elements
        )
        return f"({inner}{',' if len(elements) == 1 else ''})"

    def reference(self, node: expr.Expr, ctx=None) -> str:
        """Return the binding that denotes one value in the current definition."""
        if isinstance(node, core.ModuleList):
            return node.key
        if (
            isinstance(node, expr.Call)
            and isinstance(node.target, tensor.TupleGetItem)
            and isinstance(node.args[0], core.ModuleList)
        ):
            return self.tuple_get_item_text(node)
        if isinstance(node, expr.Tuple):
            return self.tuple_reference(node.elements, ctx)
        if id(node) in self.param_alias:
            return self.reference(self.param_alias[id(node)], ctx)
        projection = region_projection(node)
        if isinstance(projection, core.LoopRegion):
            return self.reference(projection.params[projection_index(node)], ctx)
        if isinstance(node, core.LoopRegion):
            carried = tuple(
                self.reference(carry, ctx) for carry in node.params[: len(node.yield_values)]
            )
            return carried[0] if len(carried) == 1 else "(" + ", ".join(carried) + ")"
        if isinstance(projection, core.MeshRegion):
            return self.reference(projection.body.elements[projection_index(node)], ctx)
        if isinstance(node, core.MeshRegion):
            return self.reference(node.body, ctx)
        return self.names.get(id(node)) or expr.binding_name(node) or self.visit(node, ctx)

    def tuple_get_item_text(self, node: expr.Call, ctx=None) -> str:
        held, index = node.args
        return f"{self.reference(held)}[{self.visit(index, ctx)}]"

    def slice_start(self, start, size, stride) -> str:
        moved = self.moved_window(start, size, stride)
        if moved is None:
            return repr(start.value) if isinstance(start, expr.Constant) else self.reference(start)
        window, offset, scale = moved
        name = self.reference(window)
        if scale is not None:
            name = f"{name} * {scale}"
        return moved_window_ref(name, offset)

    def visit_program_call(self, node: expr.Call, ctx=None) -> str:
        """Render one HIR call after expression-level dispatch selected it."""
        target = node.target
        args_text = ", ".join(self.reference(arg) for arg in node.args)
        if isinstance(target, core.ModuleList):
            return f"{target.key}({args_text})"
        if isinstance(target, expr.Call):
            return f"{self.reference(target)}({args_text})"
        if isinstance(target, core.Function):
            binding = module_callee_binding(target, self.child_entries)
            return f"{binding or target.name}({args_text})"
        if isinstance(target, tensor.Slice):
            starts = node.args[1]
            if not isinstance(starts, expr.Tuple):
                raise ValueError("canonical_source: Slice starts must be a Tuple")
            indexers: list[str] = []
            runtime_starts = False
            for axis, (start, size, stride) in enumerate(
                zip(starts.elements, target.sizes, target.strides)
            ):
                if self.moved_window(start, size, stride) is not None:
                    indexers.append(self.slice_start(start, size, stride))
                    continue
                dim = node.args[0].type.shape[axis]
                if isinstance(start, expr.Constant) and start.value == 0 and size == dim and stride == 1:
                    indexers.append(":")
                    continue
                if not (
                    isinstance(start, expr.Constant)
                    and isinstance(start.value, int)
                    and isinstance(size, int)
                    and isinstance(stride, int)
                ):
                    runtime_starts = True
                    break
                begin = int(start.value)
                stop = begin + size * stride
                indexers.append(f"{begin}:{stop}" if stride == 1 else f"{begin}:{stop}:{stride}")
            if runtime_starts:
                start_refs = ", ".join(
                    self.slice_start(start, size, stride)
                    for start, size, stride in zip(starts.elements, target.sizes, target.strides)
                )
                if len(starts.elements) == 1:
                    start_refs += ","
                return (
                    f"tf.slice({self.reference(node.args[0])}, ({start_refs}), "
                    f"sizes={attr_tuple_str(target.sizes, self, ctx)}, "
                    f"strides={attr_tuple_str(target.strides, self, ctx)})"
                )
            return f"{self.reference(node.args[0])}[{', '.join(indexers)}]"

        registered_name = expr.get_registered_name(target)
        dialect, *path, name = registered_name.split(".")
        selected = op_registry.lookup(dialect, tuple(path), name)
        suppressed = frozenset(selected.fixed_attrs)
        attrs = self.op_attributes(
            target,
            ctx,
            self.type_indent + "    ",
            suppressed=suppressed,
            storage=getattr(node.type, "storage", None),
        )
        schema = expr.schema_of(type(target))
        inputs = tuple(param for param in schema.signature if param.kind == "input") if schema else ()
        arguments = (
            [self.tuple_reference(node.args, ctx)]
            if len(inputs) == 1 and expr.variadic_item_annotation(inputs[0]) is not None
            else [self.reference(arg) for arg in node.args]
        )
        return f"{registered_name}({', '.join([*arguments, *attrs])})"


@dataclass(frozen=True)
class PythonPrintOptions:
    """Optional non-canonical annotations for inspection output."""

    show_types: bool = False
    comment_metadata_types: tuple[type[expr.IRMetadata], ...] = ()
    comment_opt_in: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PrintedStatement:
    """The left-hand side and physical start line of an emitted value definition."""

    value: str
    line: int


@dataclass(frozen=True)
class PythonRendering:
    """Rendered source and the Call statements produced in that same pass."""

    source: str
    statements: dict[int, PrintedStatement]


def physical_line_count(lines: list[str]) -> int:
    return sum(line.count("\n") + 1 for line in lines)


def compact_type(ty: object, printer: PythonPrinter, ctx) -> str:
    """One physical-line, DSL-shaped type annotation for inspection output."""
    if isinstance(ty, (types.TensorType, types.TupleType)):
        with printer.type_surface():
            rendered = printer.visit(ty, ctx)
        return " ".join(rendered.split())
    return repr(ty)


def metadata_comments(node: expr.Expr, options: PythonPrintOptions, printer: PythonPrinter, ctx) -> str:
    """Return same-line annotations for one printed statement.

    Omit the binding because the left-hand side already carries its emitted name
    and importing recovers it there. Emit types as annotation fragments without
    a redundant ``type=`` key.

    Part zero is the value's own type, which carries no key: it is not a
    measurement of the value, it is the value, and it is DSL text that can be
    pasted back. Every later part is what a record measured, and ``PARTS`` is
    the boundary between those two languages.
    """
    comments: list[str] = []
    if options.show_types and not isinstance(node, (core.LoopRegion, core.MeshRegion)):
        comments.append(compact_type(node.type, printer, ctx))
    for metadata_type in options.comment_metadata_types:
        metadata = expr.get_metadata(node, metadata_type)
        if metadata is None:
            continue
        comment = printer.comment(metadata, opt_in=options.comment_opt_in)
        if comment is not None:
            comments.append(comment)
    return f"  # {PARTS.join(comments)}" if comments else ""


def moved_window_ref(name: str, offset: int) -> str:
    """A tile-window indexer, carrying the compile-time offset that moves it."""
    if offset == 0:
        return name
    return f"{name} + {offset}" if offset > 0 else f"{name} - {-offset}"


def attr_tuple_str(value: tuple, printer: PythonPrinter, ctx) -> str:
    """Render an attribute's tuple value as a Python tuple literal.

    A shape-valued attribute -- `new_shape`, a tile's extents -- can hold a
    `DimVar` or dim arithmetic, and the tuple's own `str` would render those as
    dataclass reprs. Printing them the way the annotations do keeps one program
    described one way, and keeps the printed source importable: the declaration
    the header emits binds the name, not the repr.
    """
    rendered = tuple(
        printer.visit(entry, ctx) if is_dim_entry(entry) else repr(entry) for entry in value
    )
    if len(rendered) == 1:
        return f"({rendered[0]},)"
    return "(" + ", ".join(rendered) + ")"


def is_dim_entry(entry: object) -> bool:
    """Whether *entry* is a dimension rather than a plain attribute value."""
    return isinstance(entry, (expr.DimVar, expr.Var, expr.Constant, expr.Call))


def tensor_import_names(fn: core.Function) -> str:
    """``"Tensor"`` or ``"ConstTensor, Tensor"``.

    ``"Tensor"`` or ``"ConstTensor, Tensor"`` — whichever the printed
    signature (base plus every variant) actually references.
    """
    if any(p.is_const for f in (fn, *fn.variants) for p in f.params):
        return "ConstTensor, Tensor"
    return "Tensor"


def sanitize_name(name: str) -> str:
    """Make a Python-safe identifier from a loc string."""
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe or "v"


def constraint_value_str(value: object) -> str:
    if clause.is_layout_wildcard(value):
        return "_"
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", value):
        return value
    return repr(value)


def layout_constraint_str(constraint: clause.Layout) -> str:
    split_bindings = {
        attr.axis: (topology, attr)
        for topology, attr in constraint.bindings
        if isinstance(attr, types.Split)
    }
    dims: list[str] = []
    for index, extent in enumerate(constraint.layout.shape):
        if index in split_bindings:
            topology, _ = split_bindings[index]
            dims.append(f"{constraint_value_str(extent)} @ {constraint_value_str(topology)}")
        else:
            dims.append(constraint_value_str(extent))
    dims_str = "(" + ", ".join(dims) + ("," if len(dims) == 1 else "") + ")"
    bindings = [
        (topology, attr) for topology, attr in constraint.bindings if not isinstance(attr, types.Split)
    ]
    if not bindings:
        return dims_str
    binding_str = []
    for topology, attr in bindings:
        if isinstance(attr, types.Broadcast):
            binding_str.append(f"{constraint_value_str(topology)} @ B()")
        elif isinstance(attr, types.Partial):
            binding_str.append(f'{constraint_value_str(topology)} @ P("{attr.reduction}")')
        else:
            raise TypeError(f"unsupported layout binding {type(attr).__name__}")
    return f"{dims_str} @ {{{', '.join(binding_str)}}}"


def where_str(metadata: clause.WhereMetadata) -> str:
    layout = next(
        (item for item in metadata.constraints if isinstance(item, clause.Layout)),
        None,
    )
    fields: list[str] = []
    if layout is not None:
        fields.append(f"layout={layout_constraint_str(layout)}")
    for item in metadata.constraints:
        if isinstance(item, clause.Mesh):
            fields.append(f"mesh={PythonPrinter().visit(item.mesh, HirPrintContext())}")
        elif isinstance(item, clause.Storage):
            fields.append(f'storage="{item.storage.name.lower()}"')
    return "where(" + ", ".join(fields) + ")"


def constraint_line(node: expr.Expr, indent: str, name: str) -> str | None:
    metadata = clause.metadata(node)
    if metadata is None:
        return None
    return f"{indent}{name}: {where_str(metadata)}"


def iter_exprs(root: expr.Expr | None, seen: set[int] | None = None) -> Iterator[expr.Expr]:
    """Iter exprs.

    Post-order traversal of *root* and its descendants via
    ``tilefoundry.ir.core.visitor.expr_children`` (which, unlike the hand-rolled
    walkers this replaces, descends into ``LoopRegion``). Each node is
    yielded exactly once by object identity; *seen* lets callers share dedup
    state across repeated calls (e.g. one per function param).
    """
    if root is None:
        return
    if seen is None:
        seen = set()
    key = id(root)
    if key in seen:
        return
    seen.add(key)
    for child in core.expr_children(root):
        yield from iter_exprs(child, seen)
    yield root


def region_projection(node: expr.Expr) -> core.LoopRegion | core.MeshRegion | None:
    """Return the region projected by a constant-index ``TupleGetItem``."""
    if not (
        isinstance(node, expr.Call)
        and isinstance(node.target, tensor.TupleGetItem)
        and len(node.args) == 2
        and isinstance(node.args[1], expr.Constant)
        and isinstance(node.args[1].value, int)
    ):
        return None
    region = node.args[0]
    if isinstance(region, core.LoopRegion):
        return region
    if isinstance(region, core.MeshRegion) and isinstance(region.body, expr.Tuple):
        return region
    return None


def projection_index(node: expr.Call) -> int:
    """The literal field selected by a region projection."""
    index = node.args[1]
    assert isinstance(index, expr.Constant) and isinstance(index.value, int)
    return index.value


def module_callee_binding(target: core.Function, child_entries: dict[int, str]) -> str | None:
    """The attribute a call on *target* was written through, if a child's entry.

    Follows the whole chain a rebuilt function records, so a target elaborated
    for its call site is still recognised. The table is keyed by the attached
    entry's identity, never by a name, which is what keeps two attached copies
    of one source Module apart.
    """
    candidate: object = target
    seen: set[int] = set()
    while isinstance(candidate, core.Function) and id(candidate) not in seen:
        seen.add(id(candidate))
        if id(candidate) in child_entries:
            return child_entries[id(candidate)]
        candidate = valuation.origin_of(candidate)
    return None


def capture_root(value, param_alias):
    while id(value) in param_alias:
        value = param_alias[id(value)]
    return value


def window_base(start, stride, param_alias):
    """Recognize an induction window, optionally scaled and shifted by integers."""
    if (value := expr.static_dim_value(start)) is not None:
        return None, value, None
    window, offset = capture_root(start, param_alias), 0
    while isinstance(window, expr.Call) and isinstance(window.target, (expr.DimAdd, expr.DimSub)):
        left, right = window.args
        shift = expr.static_dim_value(services.normalize_dim(capture_root(right, param_alias)))
        if shift is None:
            shift = expr.static_dim_value(services.normalize_dim(capture_root(left, param_alias)))
            if shift is None or not isinstance(window.target, expr.DimAdd):
                break
            left = right
        offset += shift if isinstance(window.target, expr.DimAdd) else -shift
        window = capture_root(left, param_alias)
    if isinstance(window, expr.Var) and stride == 1:
        return window, offset, None
    if isinstance(window, expr.Call) and isinstance(window.target, expr.DimMul):
        for value, coefficient in (window.args, window.args[::-1]):
            root = capture_root(value, param_alias)
            scale = expr.static_dim_value(capture_root(coefficient, param_alias))
            if isinstance(root, expr.Var) and scale is not None and scale == stride:
                return root, offset, scale
    return None


def emit_def(
    fn: core.Function,
    def_name: str,
    ctx: HirPrintContext,
    indent: str,
    options: PythonPrintOptions,
    child_entries: dict[int, str] | None = None,
    *,
    line_offset: int = 0,
    statements: dict[int, PrintedStatement] | None = None,
) -> list[str]:
    """Render one function ``def`` block: signature + body (or ``pass`` for a prototype).

    Render one function ``def`` block: signature + body (or ``pass`` for a
    prototype). The caller prepends the decorator line(s). Each call builds its
    own SSA name scope, so a base and its variants do not share names.
    *child_entries* names the attached children this body may call.
    """
    child_entries = {} if child_entries is None else child_entries
    lines: list[str] = []
    printer = HirPrinter()
    root_mesh = fn.body.mesh if isinstance(fn.body, core.MeshRegion) and not fn.specializations else None
    if root_mesh is not None:
        ctx.push_mesh(root_mesh, "mesh")

    names: dict[int, str] = {}

    seen: set[int] = set()
    order: list[expr.Expr] = list(iter_exprs(fn.body, seen))
    for p in fn.params:
        order.extend(iter_exprs(p, seen))
    region_types = (core.LoopRegion, core.MeshRegion)
    for scope in tuple(node for node in order if isinstance(node, region_types)):
        for param in scope.params:
            order.extend(iter_exprs(param, seen))
    param_alias = {
        id(param): arg
        for scope in tuple(node for node in order if isinstance(node, region_types))
        for param, arg in scope.captures()
    }

    instance_targets = {
        id(node.target)
        for node in order
        if isinstance(node, expr.Call) and isinstance(node.target, expr.Call)
    }
    op_names_set: set[str] = set()
    for node2 in order:
        if isinstance(node2, expr.Call):
            target = node2.target
            if isinstance(target, core.Function):
                op_names_set.add(target.name)
            elif isinstance(target, expr.Op) and not expr.is_dim_op_call(node2):
                op_names_set.add(expr.get_registered_name(target).split(".")[-1])

    forced_names: dict[int, str] = {}
    tile_window_steps: dict[int, object] = {}
    collapsed_slice_ids = {
        id(node.args[0])
        for node in order
        if isinstance(node, expr.Call)
        and isinstance(node.target, tensor.Reshape)
        and len(node.args) == 1
        and isinstance(node.args[0], expr.Call)
        and isinstance(node.args[0].target, tensor.Slice)
        and isinstance(node.args[0].type, types.TensorType)
        and isinstance(node.type, types.TensorType)
        and len(node.type.shape) < len(node.args[0].type.shape)
    }

    grid_internal_ids: set[int] = set()
    mesh_region_internal_ids: set[int] = set()
    nested_grid_ids: set[int] = set()
    for node2 in order:
        if not isinstance(node2, core.LoopRegion):
            continue
        if any(
            isinstance(candidate, expr.Call)
            and isinstance(candidate.target, tensor.Slice)
            and id(candidate) not in collapsed_slice_ids
            and len(candidate.args) == 2
            and isinstance(candidate.args[1], expr.Tuple)
            and any(
                (window := window_base(start, stride, param_alias)) is not None
                and window[0] is node2.induction_var
                and size == node2.step
                for start, size, stride in zip(
                    candidate.args[1].elements,
                    candidate.target.sizes,
                    candidate.target.strides,
                )
            )
            for candidate in order
        ):
            tile_window_steps[id(node2.induction_var)] = node2.step
        for carry, init, value in zip(
            node2.params[: len(node2.yield_values)],
            node2.args[: len(node2.yield_values)],
            node2.yield_values,
        ):
            forced_names[id(carry)] = sanitize_name(carry.name)
            if not isinstance(init, expr.Var):
                forced_names[id(init)] = sanitize_name(carry.name)
        for _ in iter_exprs(node2.body, grid_internal_ids):
            pass
        for value in node2.yield_values:
            for _ in iter_exprs(value, grid_internal_ids):
                pass
        for nested in iter_exprs(node2.body, set()):
            if isinstance(nested, core.LoopRegion) and nested is not node2:
                nested_grid_ids.add(id(nested))
        for value in node2.yield_values:
            for nested in iter_exprs(value, set()):
                if isinstance(nested, core.LoopRegion) and nested is not node2:
                    nested_grid_ids.add(id(nested))

    root_grid_init_ids: set[int] = set()
    for node2 in order:
        if not isinstance(node2, core.LoopRegion) or id(node2) in nested_grid_ids:
            continue
        for init in node2.args:
            pending = [init]
            while pending:
                value = pending.pop()
                if id(value) in root_grid_init_ids:
                    continue
                root_grid_init_ids.add(id(value))
                pending.extend(value.args if isinstance(value, region_types) else core.expr_children(value))
    grid_internal_ids.difference_update(root_grid_init_ids)

    for node2 in order:
        if isinstance(node2, core.MeshRegion):
            for _ in iter_exprs(node2.body, mesh_region_internal_ids):
                pass

    def moved_window(start, size, stride):
        """The tile window and offset *start* moves it by, else ``None``."""
        window = window_base(start, stride, param_alias)
        if window is not None and tile_window_steps.get(id(window[0])) == size:
            return window
        return None

    inlined_start_ids = {
        id(start)
        for node in order
        if isinstance(node, expr.Call)
        and isinstance(node.target, tensor.Slice)
        and len(node.args) == 2
        and isinstance(node.args[1], expr.Tuple)
        for start, size, stride in zip(
            node.args[1].elements, node.target.sizes, node.target.strides
        )
        if moved_window(start, size, stride) is not None
    }
    window_start_call_ids = {
        id(node)
        for node3 in order
        if isinstance(node3, expr.Call) and isinstance(node3.target, tensor.Slice)
        and isinstance(node3.args[1], expr.Tuple)
        for start, size, stride in zip(
            node3.args[1].elements, node3.target.sizes, node3.target.strides
        )
        if moved_window(start, size, stride) is not None
        for node in iter_exprs(start, set())
        if isinstance(node, expr.Call)
    }
    tuple_index_ids: set[int] = set()
    for node2 in order:
        if isinstance(node2, expr.Call) and isinstance(node2.target, tensor.TupleGetItem):
            for nested in iter_exprs(node2.args[1], set()):
                if not isinstance(nested, expr.Var):
                    tuple_index_ids.add(id(nested))
    inlined_start_ids.update(tuple_index_ids)

    def preferred_name(node: expr.Expr) -> str | None:
        key = id(node)
        if key in forced_names:
            return forced_names[key]
        if isinstance(node, expr.Var) and node.name:
            return sanitize_name(node.name)
        if key not in window_start_call_ids and (authored_name := expr.binding_name(node)):
            return sanitize_name(authored_name)
        return None

    positions = {
        id(node): position
        for position, node in enumerate(node3 for node3 in order if not isinstance(node3, expr.Var) or not node3.name)
    }
    reserved_names = {
        name for node in order if (name := preferred_name(node)) is not None
    }

    def assign_name(node: expr.Expr) -> str:
        """Use definition positions independent of import-supplied binding names.

        Named Vars declare parameters; other values occupy definition positions.
        Reserve authored identifiers before generating names or suffixes.
        """
        key = id(node)
        if key in names:
            return names[key]
        preferred = preferred_name(node)
        name = preferred if preferred is not None else f"v{positions[key]}"
        if key in forced_names and name in names.values():
            names[key] = name
            return name

        base = name
        n = 2
        while name in names.values() or (name in reserved_names and name != preferred):
            name = f"{base}_{n}"
            n += 1
        names[key] = name
        return name

    for node2 in order:
        if id(node2) not in tuple_index_ids:
            assign_name(node2)
    for node2 in order:
        if isinstance(node2, core.LoopRegion):
            for carry in node2.params[: len(node2.yield_values)]:
                assign_name(carry)
    for node2 in order:
        if isinstance(node2, core.LoopRegion):
            assign_name(node2.induction_var)
    printer.bind_def(names, param_alias, child_entries, moved_window)
    shared = getattr(ctx, "value_printer", None) or printer
    shared.define(printer, fn, names)
    printer.share(fn, shared)

    return_type = fn.return_type
    arrow = ""
    if isinstance(return_type, types.TensorType):
        with printer.type_surface(indent=indent):
            arrow = " -> " + printer.visit(return_type, ctx)
    elif not isinstance(return_type, types.TupleType):
        arrow = " -> None"

    lines.append(f"def {def_name}(")
    params: list[str] = []
    for param in fn.params:
        name = names[id(param)]
        if isinstance(param.type, types.TensorType):
            with printer.type_surface(indent=indent, const=param.is_const):
                annotation = printer.visit(param.annotation, ctx)
            params.append(f"{indent}{name}: {annotation}")
        else:
            params.append(f"{indent}{name}")
    for index, text in enumerate(params):
        suffix = "," if index < len(params) - 1 else ""
        lines.extend((text + suffix).split("\n"))
    lines.append(f"){arrow}:")

    for param in fn.params:
        line = constraint_line(param, indent, names[id(param)])
        if line is not None:
            lines.append(line)
    if fn.body is None:
        lines.append(f"{indent}pass")
        if root_mesh is not None:
            ctx.pop_mesh()
        return lines

    printed: set[int] = {id(param) for param in fn.params}

    def emit_inline_call(node: expr.Call, level: str) -> None:
        name = names[id(node)]
        if statements is not None:
            statements[id(node)] = PrintedStatement(
                value=name,
                line=line_offset + physical_line_count(lines) + 1,
            )
        with printer.type_surface(indent=level):
            rendered = printer.visit(node, ctx)
        lines.append(f"{level}{name} = {rendered}{metadata_comments(node, options, printer, ctx)}")
        printed.add(id(node))

    def emit_expr(node: expr.Expr, level: str) -> None:
        key = id(node)
        if key in printed:
            return
        if key in instance_targets:
            for operand in core.expr_children(node):
                emit_expr(operand, level)
            printed.add(key)
            return
        if key in inlined_start_ids:
            printed.add(key)
            return
        if isinstance(node, expr.Var):
            printed.add(key)
            return
        if isinstance(node, expr.Constant):
            lines.append(
                f"{level}{names[id(node)]} = {printer.print(node.value, ctx)}"
                f"{metadata_comments(node, options, printer, ctx)}"
            )
            printed.add(key)
            return
        if isinstance(node, expr.Tuple):
            for element in node.elements:
                if not isinstance(element, expr.Constant):
                    emit_expr(element, level)
            printed.add(key)
            return
        if isinstance(node, core.LoopRegion):
            emit_loop_region(node, level)
            return
        if isinstance(node, core.MeshRegion):
            emit_mesh_region(node, level)
            return
        if isinstance(node, expr.Call):
            projection = region_projection(node)
            if isinstance(projection, core.LoopRegion):
                emit_loop_region(projection, level)
                printed.add(key)
                return
            if isinstance(projection, core.MeshRegion):
                emit_mesh_region(projection, level)
                printed.add(key)
                return
            for arg in node.args:
                emit_expr(arg, level)
            emit_inline_call(node, level)
            return
        printed.add(key)

    def emit_loop_region(region: core.LoopRegion, level: str) -> None:
        key = id(region)
        if key in printed:
            return
        for init in region.args:
            emit_expr(init, level)
        for carry, init in zip(
            region.params[: len(region.yield_values)],
            region.args[: len(region.yield_values)],
            strict=True,
        ):
            if isinstance(init, expr.Var) and names[id(carry)] != printer.reference(init):
                lines.append(f"{level}{names[id(carry)]} = {printer.reference(init)}")
            printed.add(id(carry))
        extent = printer.visit(region.extent, ctx)
        step = printer.visit(region.step, ctx)
        start = printer.visit(region.start, ctx)
        if id(region.induction_var) in tile_window_steps:
            loop = (
                f"tf.tile({extent}, {step})"
                if region.start == 0
                else f"tf.tile({start}, {extent}, {step})"
            )
        elif region.start == 0 and region.step == 1:
            loop = f"range({extent})"
        else:
            loop = f"range({start}, {extent}, {step})"
        lines.append(
            f"{level}for {names[id(region.induction_var)]} in {loop}:{metadata_comments(region, options, printer, ctx)}"
        )
        if statements is not None:
            for carry in region.params[: len(region.yield_values)]:
                statements[id(carry)] = PrintedStatement(
                    value=names[id(carry)], line=line_offset + physical_line_count(lines)
                )
        printed.add(key)
        inner = level + "    "
        emit_expr(region.body, inner)
        for value in region.yield_values:
            emit_expr(value, inner)
        for carry, value in zip(region.params[: len(region.yield_values)], region.yield_values):
            lines.append(f"{inner}{names[id(carry)]} = {printer.reference(value)}")

    def emit_mesh_region(region: core.MeshRegion, level: str, *, terminal: bool = False) -> None:
        key = id(region)
        if key in printed:
            return
        for arg in region.args:
            emit_expr(arg, level)
        if root_mesh is not None and region is fn.body:
            printed.add(key)
            emit_expr(region.body, level)
            if terminal:
                lines.append(f"{level}return {printer.reference(region.body, ctx)}")
            return
        mesh_text = printer.mesh_context(region.mesh, ctx)
        mesh_name = ctx.scope_name(region.mesh)
        lines.append(
            f"{level}with {mesh_text} as {mesh_name}:{metadata_comments(region, options, printer, ctx)}"
        )
        printed.add(key)
        inner = level + "    "
        ctx.push_mesh(region.mesh, mesh_name)
        try:
            if terminal and isinstance(region.body, core.MeshRegion):
                emit_mesh_region(region.body, inner, terminal=True)
                return
            emit_expr(region.body, inner)
            if terminal:
                lines.append(f"{inner}return {printer.reference(region.body, ctx)}")
        finally:
            ctx.pop_mesh()

    for node2 in order:
        if (
            isinstance(node2, (expr.Var, core.ModuleList))
            or id(node2) in instance_targets
            or id(node2) in grid_internal_ids
            or id(node2) in mesh_region_internal_ids
        ):
            continue
        if id(node2) in inlined_start_ids:
            printed.add(id(node2))
            continue
        if isinstance(node2, core.LoopRegion):
            emit_loop_region(node2, indent)
            continue
        if isinstance(node2, core.MeshRegion):
            emit_mesh_region(node2, indent, terminal=node2 is fn.body)
            continue
        if region_projection(node2) is not None:
            printed.add(id(node2))
            continue
        if isinstance(node2, expr.Constant):
            name = names[id(node2)]
            lines.append(
                f"{indent}{name} = {printer.print(node2.value, ctx)}{metadata_comments(node2, options, printer, ctx)}"
            )
            line = constraint_line(node2, indent, name)
            if line is not None:
                lines.append(line)
            printed.add(id(node2))
            continue
        if isinstance(node2, expr.Tuple):
            continue
        if isinstance(node2, expr.Call):
            name = names[id(node2)]
            if statements is not None:
                statements[id(node2)] = PrintedStatement(
                    value=name,
                    line=line_offset + physical_line_count(lines) + 1,
                )
            with printer.type_surface(indent=indent):
                rendered = printer.visit(node2, ctx)
            lines.append(f"{indent}{name} = {rendered}{metadata_comments(node2, options, printer, ctx)}")
            line = constraint_line(node2, indent, name)
            if line is not None:
                lines.append(line)
            printed.add(id(node2))

    if not isinstance(fn.body, core.MeshRegion):
        if isinstance(fn.body, expr.Tuple):
            lines.append(f"{indent}return {printer.tuple_reference(fn.body.elements, ctx)}")
        elif isinstance(fn.body, core.LoopRegion):
            values = tuple(
                names[id(carry)] for carry in fn.body.params[: len(fn.body.yield_values)]
            )
            result = values[0] if len(values) == 1 else "(" + ", ".join(values) + ")"
            lines.append(f"{indent}return {result}")
        else:
            body_name = printer.reference(fn.body)
            lines.append(f"{indent}return {body_name}")
    if root_mesh is not None:
        ctx.pop_mesh()
    return lines


def new_hir_context(*, target=None) -> HirPrintContext:
    """Create a HIR context with imports owned by the surrounding file."""
    ctx = HirPrintContext(target=target)
    if target is not None:
        rendered = target.to_python()
        ctx.imports.update(rendered.imports)
    return ctx


def variant_binding_name(variant: core.Function) -> str:
    """Return a valid source binding for a variant without display metadata."""
    label = valuation.display_name(variant)
    if label is not None:
        return label
    signature = valuation.canonical_specialization_signature(variant.specializations)
    return "variant_" + re.sub(r"[^0-9A-Za-z_]", "_", signature)


def emit_decorated_defs(
    fn: core.Function,
    ctx: HirPrintContext,
    indent: str,
    options: PythonPrintOptions,
    child_entries: dict[int, str] | None = None,
    *,
    line_offset: int = 0,
    statements: dict[int, PrintedStatement] | None = None,
) -> list[str]:
    """Emit decorated defs.

    Emit a base ``@func`` definition followed by one specialization block per
    variant. Standalone and module output share this path so dispatch prototypes
    render identically.
    See [inspection §2.6](docs/spec/inspection.md#26-specialization-printing).
    """
    printer = HirPrinter()
    decorator = "@func"
    if isinstance(fn.body, core.MeshRegion) and not fn.specializations:
        root_ctx = HirPrintContext()
        with printer.type_surface(indent=indent):
            mesh_text = printer.visit(fn.body.mesh, root_ctx)
        ctx.imports.update(root_ctx.imports)
        decorator = f"@func(mesh={mesh_text})"
    lines: list[str] = [decorator]
    lines.extend(
        emit_def(
            fn,
            fn.name,
            ctx,
            indent,
            options,
            child_entries,
            line_offset=line_offset + physical_line_count(lines),
            statements=statements,
        )
    )

    for variant in fn.variants:
        lines.append("")
        lines.append(
            f"@{fn.name}.specialize({HirPrinter().render_pattern(variant.specializations[0], ctx)})"
        )
        lines.extend(
            emit_def(
                variant,
                variant_binding_name(variant),
                ctx,
                indent,
                options,
                child_entries,
                line_offset=line_offset + physical_line_count(lines),
                statements=statements,
            )
        )
    return lines


def render_hir_function(
    fn: core.Function,
    *,
    options: PythonPrintOptions | None = None,
    printer: HirPrinter | None = None,
    target=None,
) -> PythonRendering:
    """Render a HIR Function and locate every Call equation in the same pass.

    A normal function prints as a single ``@func``. A dispatch prototype
    (``variants != ()``) prints as a ``pass``-bodied ``@func`` base followed by
    one ``@<name>.specialize(pattern)`` block per variant. Placement sugar is
    emitted only where an explicit mesh-scope binding dominates its use.
    """
    indent = "    "
    ctx = HirPrintContext(target=target, value_printer=printer or HirPrinter())
    statements: dict[int, PrintedStatement] = {}
    lines = emit_decorated_defs(
        fn,
        ctx,
        indent,
        options or PythonPrintOptions(),
        line_offset=0,
        statements=statements,
    )
    header = ctx.header()
    header_lines = physical_line_count(header)
    statements = {
        identity: PrintedStatement(value=item.value, line=item.line + header_lines)
        for identity, item in statements.items()
    }
    ctx.value_printer.add_statements(statements)
    return PythonRendering("\n".join(header + lines) + "\n", statements)


def hir_function_to_python(
    fn: core.Function,
    *,
    options: PythonPrintOptions | None = None,
) -> str:
    """Convert a HIR Function to canonical Python DSL source."""
    return HirPrinter().print(fn, options=options)


def as_script(
    fn: core.Function | core.PrimFunction | core.Module,
    *,
    module: str | None = None,
    options: PythonPrintOptions | None = None,
) -> str:
    """Convert an HIR function or module to Python DSL source.

    Without *module*, emit a standalone decorated function. With *module*, emit
    a named module-class wrapper with entry and mesh definitions. *options*
    controls canonical-source rendering.
    """
    if isinstance(fn, core.Module):
        if fn.functions and all(isinstance(item, core.PrimFunction) for item in fn.functions):
            return tir_module_to_python(fn, module)
        return module_source(fn, module, options=options)
    if isinstance(fn, core.PrimFunction) and module is None:
        return tir_function_to_python(fn)
    if module is not None:
        if isinstance(fn, core.PrimFunction):
            return tir_module_to_python(
                core.Module(name=module, functions=(fn,), entry=fn.name)
            )
        return module_source(fn, module, options=options)
    return hir_function_to_python(fn, options=options)


def module_to_python(fn: core.Function, module_name: str = "M") -> str:
    """Backward-compat alias for ``as_script(fn, module=module_name)``."""
    return as_script(fn, module=module_name)


def emission_order(mod: core.Module) -> tuple:
    """A Module's functions in the order the printed class body binds them.

    The entry goes last: a body calling a sibling names the attribute the class
    body already bound, so every callee must be written before it. Mesh
    traversal reads the same order, so output does not depend on the order the
    authored source happened to use.
    """
    functions = mod.functions
    entry = mod.entry_function() if functions and mod.entry is not None else None
    ordered = tuple(fn for fn in functions if fn is not entry)
    return ordered + (entry,) if entry is not None else ordered


def module_decorator_line(mod: core.Module, entry_name: str | None, ctx: HirPrintContext) -> str:
    """Render the context this Module declares as an ``@module(...)`` line.

    Always the called form. A bare decorator has not run while the class body
    is evaluated, so a body naming a child call could not resolve it.
    """
    kwargs: list[str] = [] if entry_name is None else [f'entry="{entry_name}"']
    printer = HirPrinter()
    if mod.target is not None:
        rendered: utils.PythonExpr = mod.target.to_python()
        ctx.imports.update(rendered.imports)
        kwargs.append(f"target={rendered.text}")
    if mod.topologies is not None:
        topo_strs = [f'Topology("{t.name}", {printer.visit(t.size, ctx)})' for t in mod.topologies]
        rendered_topologies = f"({', '.join(topo_strs)},)" if topo_strs else "()"
        kwargs.append(f"topologies={rendered_topologies}")
    return f"@module({', '.join(kwargs)})"


def emit_module_class(
    mod: core.Module,
    module_name: str,
    ctx: HirPrintContext,
    indent: str,
    options: PythonPrintOptions,
    names_by_id,
) -> list[str]:
    """One ``@module`` class block: its instance slots, then its functions.

    Children first, because a body calling one names the attribute it is bound
    to and a class body binds in the order it is written.
    """
    previous = ctx.enter_module(mod.target, mod.topologies)
    lines = [module_decorator_line(mod, mod.entry, ctx), f"class {module_name}:"]
    ordered = emission_order(mod)
    child_entries = {}
    bindings = [slot_binding(key, slot, names_by_id) for key, slot in mod.children.items()]
    blocks: list[list[str]] = [bindings] if bindings else []
    for fn in ordered:
        if isinstance(fn, core.Function):
            blocks.append(emit_decorated_defs(fn, ctx, indent, options, child_entries))
        elif isinstance(fn, core.PrimFunction):
            tir_block = function_block(fn, frame=valuation.Context(mod, fn))
            ctx.imports.update(tir_block.imports)
            blocks.append(tir_block)
        else:
            raise TypeError(f"Python printer cannot serialize {type(fn).__name__}")
    for index, block in enumerate(blocks):
        if index:
            lines.append("")
        lines.extend(f"{indent}{ln}" if ln else ln for ln in block)
    ctx.leave_module(previous)
    return lines


def module_source(
    fn_or_module: core.Function | core.Module,
    module_name: str | None = None,
    *,
    options: PythonPrintOptions | None = None,
) -> str:
    """Render a function or a whole Module tree as ``@module`` source."""
    if isinstance(fn_or_module, core.Module):
        root = fn_or_module
        module_name = root.name if module_name is None else module_name
    else:
        root = core.Module(
            name="M" if module_name is None else module_name,
            functions=(fn_or_module,),
            entry=fn_or_module.name,
        )
        module_name = root.name
    functions = tuple(fn for node in module_definition_order(root) for fn in node.functions)
    if not functions:
        raise TypeError("Module printer requires at least one function")
    entry = root.entry_function() if root.entry is not None else None
    if entry is not None and not isinstance(entry, (core.Function, core.PrimFunction)):
        raise TypeError("Module printer requires a function entry")

    indent4 = "    "
    ctx = new_hir_context(target=root.target)
    names = definition_names(root, module_name)
    lines = []
    for node in module_definition_order(root):
        if lines:
            lines.append("")
        lines.extend(
            emit_module_class(
                node, names[id(node)], ctx, indent4, options or PythonPrintOptions(), names
            )
        )
        if node is not root and names[id(node)] != node.name:
            lines.append(f"{names[id(node)]}.name = {node.name!r}")
    if root.children:
        lines.extend(
            (
                "",
                f"del {', '.join(names[id(node)] for node in module_definition_order(root) if node is not root)}",
            )
        )
    header = ctx.header()
    return "\n".join(header + lines) + "\n"


__all__ = [
    "HirPrinter",
    "PythonPrintOptions",
    "as_script",
    "hir_function_to_python",
    "module_to_python",
]
