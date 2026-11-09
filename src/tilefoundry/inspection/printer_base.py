"""Canonical Python visitor shared by the HIR and TIR printers."""

from __future__ import annotations

import enum
import json
import math
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from types import MappingProxyType

from tilefoundry import target, utils
from tilefoundry.ir import core, dim, expr, pattern, types
from tilefoundry.ir import utils as ir_utils
from tilefoundry.ir.expr import op_registry
from tilefoundry.ir.hir import sharding, tensor

PAIR, PER_UNIT, ENTRY, ENTRIES, FIELD, FIELDS, PARTS, TRIPS = (
    "/",
    "@",
    ":",
    ",",
    "=",
    " ",
    "; ",
    "*",
)

UNSET = object()


DIM_INFIX_OPS = MappingProxyType({
    expr.DimAdd: "+",
    expr.DimSub: "-",
    expr.DimMul: "*",
    expr.DimFloorDiv: "//",
    expr.DimMod: "%",
})

DIM_FUNC_OPS = MappingProxyType({
    expr.DimMin: "min",
    expr.DimMax: "max",
})


def empty_without_default(value):
    if isinstance(value, Mapping | tuple | list):
        return not value
    return False


def ceildiv_operands(value: expr.Call) -> tuple[object, object] | None:
    """Recover the public constructor from ceildiv's canonical arithmetic tree."""
    if not isinstance(value.target, expr.DimFloorDiv) or len(value.args) != 2:
        return None
    numerator, divisor = value.args
    if not (
        isinstance(numerator, expr.Call)
        and isinstance(numerator.target, expr.DimSub)
        and len(numerator.args) == 2
        and isinstance(numerator.args[1], expr.Constant)
        and numerator.args[1].value == 1
    ):
        return None
    added = numerator.args[0]
    if not (
        isinstance(added, expr.Call)
        and isinstance(added.target, expr.DimAdd)
        and len(added.args) == 2
        and added.args[1] == divisor
    ):
        return None
    return added.args[0], divisor


class PythonPrinter(core.PrinterBase, expr.Functor[str], core.TypeFunctor[str]):
    """Render the Python DSL with one dispatch root for expressions and types."""

    def print_SSAValue(self, value):
        return self.print(value)

    def comment(self, record, *, opt_in=frozenset()) -> str | None:
        method = getattr(self, f"print_{type(record).__name__}", None)
        return (method(record, opt_in=opt_in) or None) if method else None

    def comment_value(self, value):
        method = getattr(self, f"print_{type(value).__name__}", None)
        if method:
            return method(value)
        if is_dataclass(value):
            return ENTRIES.join(
                f"{item.name}{ENTRY}{self.comment_value(getattr(value, item.name))}"
                for item in fields(value)
            )
        if isinstance(value, Mapping):
            return ";".join(f"{k}{ENTRY}{self.comment_value(v)}" for k, v in value.items())
        if isinstance(value, (tuple, list)):
            return ENTRIES.join(self.comment_value(v) for v in value)
        return str(value)

    def print_Prose(self, value):
        return json.dumps(str(value))

    def print_Precision(self, value):
        return value.value

    def print_TrafficBytes(self, value):
        return f"r{utils.format_bytes(value.read)}{PAIR}w{utils.format_bytes(value.write)}"

    def print_TripInterval(self, value):
        if value.trips <= 1:
            return f"[{value.start},{value.end})"
        offset = f"{value.stride}t+"
        return f"[{offset}{value.start},{offset}{value.end}){TRIPS}{value.trips}"


    def record(self, family, values):
        out = []
        for item in values:
            key, value = item[:2]
            default = item[2] if len(item) == 3 else UNSET
            if default is not UNSET:
                if value == default:
                    continue
            elif empty_without_default(value):
                continue
            out.append(f"{key.replace('_', '-')}={self.comment_value(value)}")
        if not family:
            return FIELDS.join(out)
        return FIELDS.join([family, *out]) if out else ""

    def single(self, family, value, default=UNSET):
        if default is not UNSET and value == default:
            return ""
        if default is UNSET and empty_without_default(value):
            return ""
        return f"{family}{FIELD}{self.comment_value(value)}"

    def spread(self, spread, topologies, *, logical=True):
        values = []
        if logical:
            values.append(f"{self.comment_value(spread.logical)}@logical")
        values.append(f"{self.comment_value(spread.total)}@total")
        values.extend(
            f"{self.comment_value(value)}@{topology}"
            for topology, value in zip(topologies, spread.per_unit, strict=False)
        )
        return self.comment_value(values)

    def breakdown(self, held, topologies, *, logical=True):
        return {
            kind: self.spread(spread, topologies, logical=logical) for kind, spread in held.kinds
        }

    def footprint(self, footprint):
        if footprint is None:
            return {}
        return {
            "/".join((*ownership, self.comment_value(value))): utils.format_bytes(sum(spread.logical for _, spread in breakdown.kinds))
            for value, ownership, breakdown in footprint.buffers
        }

    def print_ComputeCostMetadata(self, record, **_):
        return self.record(
            "compute-cost",
            (
                ("flops", self.breakdown(record.flops, record.topologies)),
                ("other_ops", self.breakdown(record.other_ops, record.topologies)),
                ("precision", record.precision),
            ),
        )

    def print_MemoryMetadata(self, record, *, opt_in=frozenset()):
        traffic = self.breakdown(record.traffic.storage, record.topologies)
        values = [("traffic", traffic), ("footprint", self.footprint(record.footprint))]
        if record.footprint is not None:
            values.append(("footprint_precision", record.footprint.precision))
        if "operands" in opt_in:
            last = len(record.operands) - 1
            operands = {
                "result" if index == last else str(index): moved
                for index, moved in enumerate(record.operands)
            }
            values.append(("operands", operands))
        return self.record("memory", values)

    def print_RegionMemoryMetadata(self, record, **_):
        return self.record(
            "memory",
            (
                ("traffic", self.breakdown(record.traffic.storage, record.topologies)),
                ("footprint", self.footprint(record.footprint)),
                (
                    "footprint_precision",
                    record.footprint.precision if record.footprint else None,
                    None,
                ),
                (
                    "peak",
                    {item.memory_level: utils.format_bytes(item.peak_bytes) for item in record.peaks},
                ),
                (
                    "persistent",
                    {
                        item.memory_level: utils.format_bytes(item.persistent_bytes)
                        for item in record.peaks
                        if item.persistent_bytes
                    },
                ),
            ),
        )

    def print_ReuseWindow(self, record, **_):
        return self.record(
            "",
            (
                ("buffer", "/".join((*record.ownership, self.comment_value(record.buffer)))),
                ("holds", utils.format_bytes(record.holds_bytes)),
                ("time", record.time or "none"),
                ("space", record.space or "none"),
                (
                    "reuse",
                    "unknown" if record.reuse_bytes is None else utils.format_bytes(record.reuse_bytes),
                ),
                ("fits", "yes" if record.fits else "no"),
                ("precision", record.precision),
            ),
        )

    def print_RooflineMetadata(self, record, **_):
        return self.record(
            "roofline", (("ideal_ns", record.ideal_ns, 0), ("bound_by", record.bound_by, "none"))
        )

    def print_PerformanceMetadata(self, record, **_):
        interval = core.TripInterval(
            record.timeline.start_ns,
            record.timeline.end_ns,
            record.timeline.stride_ns,
            record.timeline.trips,
        )
        return self.single("performance", interval)

    def print_PerformanceSummaryMetadata(self, record, **_):
        predicted = record.timeline.end_ns - record.timeline.start_ns
        wave = "/".join(map(str, record.wave))
        return self.record("performance", (("predicted_ns", predicted), ("wave", wave)))

    def print_SourceSpanMetadata(self, record, **_):
        return self.single("source", f"{record.file}:{record.line}:{record.column}")

    def print_ReportIdentity(self, record, **_):
        return self.record(
            "analysis",
            (
                ("target", record.target, ""),
                ("module", record.module, ""),
                ("function", record.function, ""),
                ("topology", record.topology, "none"),
            ),
        )

    def print_ReportSelection(self, record, **_):
        return self.record(
            "selection", (("requested", record.requested, ()), ("executed", record.executed, ()))
        )

    def print_AdvisorySummary(self, record, **_):
        return self.single("advisory", record.text)

    def print_ErrorSummary(self, record, **_):
        return self.single("error", record.text)

    def print_PerformanceSummaryView(self, record, **_):
        return self.record(
            "performance",
            (
                ("root", record.root, ""),
                ("predicted_ns", record.predicted_ns, 0),
                ("wave", record.wave, ""),
            ),
        )

    def __init__(self) -> None:
        expr.Functor.__init__(self)
        self.type_indent = ""
        self.tensor_head = "Tensor"
        self.in_nested_dim = False

    def visit(self, value, ctx=None):
        """Dispatch every implemented functor family by the concrete node name."""
        method = getattr(self, f"visit_{type(value).__name__}", None)
        if method is not None:
            return method(value, ctx)
        return self.default_visit(value, ctx)

    def default_visit(self, value, ctx=None) -> str:
        if isinstance(value, bool):
            return repr(value)
        if isinstance(value, float):
            return self.print(value, ctx)
        if isinstance(value, int):
            return str(value)
        raise NotImplementedError(f"no Python visit routine for {type(value).__name__}")

    @contextmanager
    def type_surface(self, *, indent: str | None = None, const: bool = False):
        """Carry statement indentation and parameter const-ness through type visits."""
        previous = (self.type_indent, self.tensor_head)
        if indent is not None:
            self.type_indent = indent
        self.tensor_head = "ConstTensor" if const else "Tensor"
        try:
            yield
        finally:
            self.type_indent, self.tensor_head = previous

    @contextmanager
    def nested_dim(self, nested: bool):
        previous = self.in_nested_dim
        self.in_nested_dim = nested
        try:
            yield
        finally:
            self.in_nested_dim = previous

    def dim_entry(self, value, ctx=None, *, nested: bool = False) -> str:
        """One entry of a shape or stride tuple, which may itself be a group.

        A layout groups the modes of one tensor axis by writing them as a
        tuple in that axis's place, so an entry is read as the shape tuple it
        is rather than as a single dimension.
        """
        if isinstance(value, tuple):
            entries = ", ".join(self.dim_entry(item, ctx) for item in value)
            return f"({entries}{',' if len(value) == 1 else ''})"
        with self.nested_dim(nested):
            return self.visit(value, ctx)

    def shape_tuple(self, shape: tuple, ctx=None) -> str:
        values = tuple(self.dim_entry(entry, ctx) for entry in shape)
        return f"({values[0]},)" if len(values) == 1 else "(" + ", ".join(values) + ")"

    def dtype_str(self, dtype: expr.DType, ctx=None) -> str:
        return dtype.name

    def visit_DimVar(self, value: expr.DimVar, ctx=None) -> str:
        if ctx is not None:
            ctx.declare_dim(value.name, value)
        return value.name

    def visit_Var(self, value: expr.Var, ctx=None) -> str:
        return value.name

    def visit_Constant(self, value: expr.Constant, ctx=None) -> str:
        return self.print(value.value, ctx)

    def visit_Tuple(self, value: expr.Tuple, ctx=None) -> str:
        rendered = ", ".join(self.visit(item, ctx) for item in value.elements)
        return f"({rendered}{',' if len(value.elements) == 1 else ''})"

    def visit_Call(self, value: expr.Call, ctx=None) -> str:
        ceildiv_args = ceildiv_operands(value)
        if ceildiv_args is not None:
            left, right = ceildiv_args
            return f"ceildiv({self.dim_entry(left, ctx)}, {self.dim_entry(right, ctx)})"
        tgt = value.target
        if isinstance(tgt, tensor.TupleGetItem):
            return self.tuple_get_item_text(value, ctx)
        if isinstance(tgt, dim.MeshCoord):
            return self.mesh_coordinate_text(value, tgt, ctx)
        if isinstance(tgt, expr.DimConst):
            return str(tgt.value)
        for op_type, symbol in DIM_INFIX_OPS.items():
            if isinstance(tgt, op_type):
                left, right = value.args
                rendered = (
                    f"{self.dim_entry(left, ctx, nested=True)} {symbol} "
                    f"{self.dim_entry(right, ctx, nested=True)}"
                )
                return f"({rendered})" if self.in_nested_dim else rendered
        for op_type, name in DIM_FUNC_OPS.items():
            if isinstance(tgt, op_type):
                args = ", ".join(self.dim_entry(arg, ctx) for arg in value.args)
                return f"{name}({args})"
        return self.visit_program_call(value, ctx)

    def tuple_get_item_text(self, value: expr.Call, ctx=None) -> str:
        held, index = value.args
        return f"{self.visit(held, ctx)}[{self.visit(index, ctx)}]"

    def mesh_coordinate_text(self, value: expr.Call, tgt: dim.MeshCoord, ctx) -> str:
        """Render one coordinate through the active binding of its mesh."""
        ctx = ctx if ctx is not None else self.context
        alias = ctx.mesh_alias(tgt.mesh)
        if value.args and isinstance(value.args[0], expr.Tuple):
            if alias is None:
                raise ValueError("MeshCoord mesh has no active binding to print")
            return f"tf.meshcoord({self.visit(value.args[0], ctx)}, mesh={alias})"
        axis = expr.static_dim_value(value.args[0]) if value.args else None
        if axis is None or axis < 0 or axis >= len(types.flatten(tgt.mesh.layout).shape):
            raise ValueError("MeshCoord requires a literal in-range axis to print")
        ref = ctx.mesh_axis_alias(tgt.mesh, axis)
        if ref is not None:
            return ref
        if alias is None:
            raise ValueError("MeshCoord mesh has no active binding to print")
        if axis < len(tgt.mesh.names):
            axis_name = tgt.mesh.names[axis]
        elif axis < 3:
            axis_name = ("x", "y", "z")[axis]
        else:
            raise ValueError("unnamed MeshCoord axes above z cannot be printed")
        return f"{alias}.{axis_name}"

    def visit_program_call(self, value: expr.Call, ctx=None) -> str:
        raise NotImplementedError(f"{type(self).__name__} cannot render program calls")


    def shard_surface(self, value: types.ShardLayout, ctx=None, *, storage=None) -> str | None:
        """Render placement sugar only when every mesh axis has a scope binding.

        The sugar states one extent per tensor axis, each a dimension
        expression, so it declines a layout whose modes are grouped by tile
        axis: writing the groups in would emit a line the parser refuses.
        """
        value = ir_utils.participant_layout(value)
        layout = value.layout
        operand = None
        if isinstance(layout, types.ComposedLayout) and isinstance(layout.outer, types.Layout):
            operand = "(" + self.layout_surface(layout, ctx) + ")"
        elif isinstance(layout, types.Layout):
            operand = self.layout_surface(layout, ctx)
        if ctx is not None and all(isinstance(attr, types.Broadcast) for attr in value.attrs):
            current = ctx.current_mesh
            if current is not None and operand is not None:
                current = ir_utils.participant_layout(types.ShardLayout(layout, tuple(
                    types.Broadcast() for _ in range(len(types.flatten(current.layout).shape))
                ), current)).mesh
                if current == value.mesh:
                    return operand + " @ {B()}"
        names = value.mesh.names
        if (
            (not isinstance(layout, types.Layout) and operand is None)
            or not names
            or len(value.attrs) != len(names)
            or ctx is None
        ):
            return None
        if isinstance(layout, types.Layout) and any(
            isinstance(entry, tuple) for entry in (*layout.shape, *(layout.strides or ()))
        ):
            return None
        refs = tuple(ctx.mesh_axis_alias(value.mesh, index) for index in range(len(names)))
        if any(ref is None for ref in refs):
            return None

        splits: dict[int, list[str]] = {}
        partials: list[str] = []
        for attr, ref in zip(value.attrs, refs, strict=True):
            assert ref is not None
            if isinstance(attr, types.Split):
                if isinstance(layout, types.ComposedLayout) or attr.axis >= len(layout.shape):
                    return None
                splits.setdefault(attr.axis, []).append(ref)
            elif isinstance(attr, (types.Partial, types.Varying)):
                partials.append(f"{ref} @ {self.visit(attr, ctx)}")
            elif not isinstance(attr, types.Broadcast):
                return None
        if not splits and not partials:
            return None
        if isinstance(layout, types.ComposedLayout):
            return operand + " @ {" + ", ".join(partials) + "}"
        states = partials

        explicit = True
        if storage is not None and ctx is not None and ctx.target is not None:
            default = types.ShardLayout.from_modes(
                value.layout.shape,
                value.mesh,
                value.attrs,
                storage=storage,
                owners=ctx.target.storage_owners,
            )
            explicit = layout != default.layout
        if explicit and any(
            axis in splits and self.dim_entry(dim_value, ctx, nested=True) != self.dim_entry(dim_value, ctx)
            for axis, dim_value in enumerate(layout.shape)
        ):
            return None

        dims = [
            (
                f"{self.dim_entry(dim_value, ctx, nested=True)} "
                + " ".join(f"@ {ref}" for ref in splits[axis])
            )
            if axis in splits
            else self.dim_entry(dim_value, ctx)
            for axis, dim_value in enumerate(layout.shape)
        ]
        dims_text = ", ".join(dims) + ("," if len(dims) == 1 else "")
        parts = [f"({dims_text})"]
        if explicit:
            parts.append(self.shape_tuple(layout.strides, ctx))
        operand = parts[0] if len(parts) == 1 else "(" + ", ".join(parts) + ")"
        if states:
            operand += " @ {" + ", ".join(states) + "}"
        return operand

    def visit_TensorType(self, value: types.TensorType, ctx=None) -> str:
        result = (
            f"{self.tensor_head}["
            f'{self.shape_tuple(value.shape, ctx)}, "{self.dtype_str(value.dtype, ctx)}"'
        )
        if isinstance(value.layout, types.ShardLayout):
            surface = self.shard_surface(value.layout, ctx, storage=value.storage)
            if surface is not None:
                result += f", {surface}"
            else:
                with self.type_surface(indent=self.type_indent + "    "):
                    result += f",\n{self.type_indent}{self.visit(value.layout, ctx)}"
        elif value.layout is not None and value.layout != types.Layout(value.shape):
            result += f", {self.layout_surface(value.layout, ctx)}"
        if value.storage is not expr.StorageKind.GMEM:
            result += f', "{value.storage.name.lower()}"'
        return result + "]"

    def visit_TupleType(self, value: types.TupleType, ctx=None) -> str:
        return f"Tuple[{', '.join(self.visit(field, ctx) for field in value.fields)}]"

    def visit_PointerType(self, value: types.PointerType, ctx=None) -> str:
        return f"PointerType(DType.{value.dtype.name}, StorageKind.{value.storage.name})"

    def visit_UnitType(self, value: types.UnitType, ctx=None) -> str:
        return "None"

    def visit_DType(self, value: expr.DType, ctx=None) -> str:
        return self.dtype_str(value, ctx)

    def visit_Mesh(self, value: types.Mesh, ctx=None) -> str:
        if ctx is not None:
            alias = ctx.mesh_alias(value)
            if alias is not None:
                return alias
            sliced = ctx.mesh_slice(value)
            if sliced is not None:
                return sliced
        return self.mesh_text(value, ctx)

    def mesh_context(self, value: types.Mesh, ctx) -> str:
        """Render a with header, including a recoverable lexical selection."""
        alias = ctx.mesh_alias(value) or ctx.mesh_slice(value)
        if alias is not None:
            return alias
        selection = ctx.mesh_selection(value)
        if selection is not None:
            selection, layout = selection
            shape = self.shape_tuple(layout.shape, ctx)
            if layout.strides != types.compact_row_major(layout.shape):
                shape = f"({shape}, {self.shape_tuple(layout.strides, ctx)})"
            names = ", ".join(json.dumps(name) for name in value.names)
            names = f"({names}{',' if len(value.names) == 1 else ''})"
            return f"Mesh({selection}, layout={shape}, names={names})"
        return self.mesh_text(value, ctx, layout_position=True)

    def mesh_text(self, value: types.Mesh, ctx=None, *, layout_position: bool = False) -> str:
        topologies = ", ".join(
            json.dumps(topology.name)
            if layout_position and ctx is not None and topology in ctx.topologies
            else f'Topology("{topology.name}", {self.dim_entry(topology.size, ctx)})'
            for topology in value.topologies
        )
        topologies = f"({topologies}{',' if len(value.topologies) == 1 else ''})"
        written = ir_utils.device_layout(value)
        if isinstance(value.layout, types.ComposedLayout):
            written = types.ComposedLayout(
                inner=value.layout.inner, offset=value.layout.offset, outer=written
            )
        if isinstance(written, types.Layout) and written.strides == types.compact_row_major(written.shape):
            rendered = self.shape_tuple(written.shape, ctx)
        else:
            rendered = self.layout_surface(written, ctx) if layout_position else self.visit(written, ctx)
        result = f"Mesh({topologies}, layout={rendered}"
        if value.names:
            names = ", ".join(json.dumps(name) for name in value.names)
            result += f", names=({names}{',' if len(value.names) == 1 else ''})"
        return result + ")"

    def visit_NoneType(self, value: None, ctx=None) -> str:
        return "None"

    def visit_Layout(self, value: types.Layout, ctx=None) -> str:
        return (
            f"Layout({self.shape_tuple(value.shape, ctx)}, {self.shape_tuple(value.strides, ctx)})"
        )

    def visit_Swizzle(self, value: types.Swizzle, ctx=None) -> str:
        return f"Swizzle({value.bits}, {value.base}, {value.shift})"

    def layout_surface(self, value: types.LayoutBase, ctx=None, *, storage=None) -> str:
        """Use composition operators only where the parser expects a layout."""
        if isinstance(value, types.ShardLayout):
            return self.shard_surface(value, ctx, storage=storage) or self.visit(value, ctx)
        if isinstance(value, types.Layout) and not any(
            isinstance(entry, tuple) for entry in (*value.shape, *value.strides)
        ):
            shape = self.shape_tuple(value.shape, ctx)
            if value.strides == types.Layout(value.shape).strides:
                return shape
            return f"({shape}, {self.shape_tuple(value.strides, ctx)})"
        if not isinstance(value, types.ComposedLayout) or not isinstance(value.outer, types.Layout):
            return self.visit(value, ctx)
        result = self.layout_surface(value.outer, ctx)
        if value.offset != 0:
            result += " + " + self.dim_entry(value.offset, ctx, nested=True)
        if value.inner is not None:
            result += " | " + self.print(value.inner, ctx, self.type_indent)
        return result

    def op_attributes(
        self,
        value: expr.Op,
        ctx=None,
        indent: str = "",
        *,
        suppressed: frozenset[str] = frozenset(),
        storage=None,
    ) -> list[str]:
        """Render attribute ParamDefs in declaration order, omitting defaults."""
        attrs = []
        for param in expr.schema_of(type(value)).signature:
            if param.kind != "attribute" or param.name in suppressed:
                continue
            attr = getattr(value, param.name, None)
            if param.default is not expr.MISSING and attr == param.default:
                continue
            with self.type_surface(indent=indent):
                if isinstance(value, sharding.Reshard) and isinstance(attr, types.ShardLayout):
                    rendered = self.shard_surface(
                        attr, ctx, storage=storage or value.storage
                    ) or self.visit(attr, ctx)
                else:
                    rendered = (
                        self.layout_surface(attr, ctx)
                        if isinstance(attr, types.LayoutBase)
                        else self.print(attr, ctx, indent)
                    )
            attrs.append(f"{param.name}={rendered}")
        return attrs

    def op_value(self, value: expr.Op, ctx=None, indent: str = "") -> str:
        """Render a schema-backed op value through its attribute values."""
        name = expr.get_registered_name(value)
        dialect, *path, leaf = name.split(".")
        schema = op_registry.lookup(dialect, tuple(path), leaf)
        attrs = self.op_attributes(value, ctx, indent, suppressed=frozenset(schema.fixed_attrs))
        return f"{name}({', '.join(attrs)})"

    def visit_ComposedLayout(self, value: types.ComposedLayout, ctx=None) -> str:
        outer, child = self.type_indent, self.type_indent + "    "
        with self.type_surface(indent=child):
            inner_text = self.visit(value.inner, ctx)
            outer_text = self.visit(value.outer, ctx)
        return (
            "ComposedLayout(\n"
            f"{child}inner={inner_text},\n"
            f"{child}offset={self.dim_entry(value.offset, ctx)},\n"
            f"{child}outer={outer_text},\n"
            f"{outer})"
        )

    def visit_ShardLayout(self, value: types.ShardLayout, ctx=None) -> str:
        value = ir_utils.participant_layout(value)
        outer, child = self.type_indent, self.type_indent + "    "
        attrs = ", ".join(self.visit(attr, ctx) for attr in value.attrs)
        if len(value.attrs) == 1:
            attrs += ","
        with self.type_surface(indent=child):
            layout_text = self.visit(value.layout, ctx)
            mesh_text = (
                ctx.mesh_alias(value.mesh) if ctx is not None else None
            ) or self.mesh_text(value.mesh, ctx)
        return (
            "ShardLayout(\n"
            f"{child}layout={layout_text},\n"
            f"{child}attrs=({attrs}),\n"
            f"{child}mesh={mesh_text},\n"
            f"{outer})"
        )

    def visit_Varying(self, value: types.Varying, ctx=None) -> str:
        return "V()"

    def visit_Broadcast(self, value: types.Broadcast, ctx=None) -> str:
        return "B()"

    def visit_Split(self, value: types.Split, ctx=None) -> str:
        return f"S({value.axis})"

    def visit_Partial(self, value: types.Partial, ctx=None) -> str:
        return f'P("{value.reduction}")'

    def print(self, value, ctx=None, indent: str = "") -> str:
        """*value* as canonical Python source: the one entry callers print through.

        A ``Printable`` writes itself with this printer; IR nodes and types go
        through the visitor, and statement lines are joined into one text.
        """
        if isinstance(value, expr.DType):
            return json.dumps(value.name)
        if isinstance(value, (types.TensorType, types.PointerType, types.Mesh, types.LayoutBase)):
            with self.type_surface(indent=indent):
                return self.visit(value, ctx)
        if not isinstance(value, enum.Enum) and dim.is_dimension(value):
            return self.visit(value, ctx)
        if isinstance(value, expr.Op):
            return self.op_value(value, ctx, indent)
        if isinstance(value, core.Printable):
            return value.print(self, ctx)
        if isinstance(value, expr.StorageKind):
            return json.dumps(value.name.lower())
        if isinstance(value, enum.Enum):
            try:
                return f"{expr.get_registered_name(value)}.{value.name}"
            except ValueError:
                pass
            if ctx is not None and type(value).__module__ != "tilefoundry.ir.expr.kinds":
                ctx.use(
                    utils.PythonExpr(
                        (f"from {type(value).__module__} import {type(value).__name__}",), ""
                    )
                )
            return f"{type(value).__name__}.{value.name}"
        if isinstance(value, target.Target):
            rendered = value.to_python()
            if ctx is not None:
                ctx.imports.update(rendered.imports)
            return rendered.text
        if isinstance(value, tuple):
            rendered = ", ".join(self.print(item, ctx, indent) for item in value)
            return f"({rendered}{',' if len(value) == 1 else ''})"
        if isinstance(value, float):
            if math.isinf(value):
                return "-1e999" if value < 0 else "1e999"
            if math.isnan(value):
                return ctx.use(utils.PythonExpr(("import math",), "math.nan")) if ctx is not None else "math.nan"
        if isinstance(value, str):
            return json.dumps(value)
        if value is None or isinstance(value, (int, float, bool)):
            return repr(value)
        printed = self.visit(value, ctx)
        return "\n".join(printed) if isinstance(printed, list) else printed

    def render_pattern(self, declared: pattern.Pattern, ctx=None) -> str:
        if isinstance(declared, pattern.Range):
            return f'pattern.Range("{declared.dim_var}", {declared.lo}, {declared.hi})'
        return repr(declared)


__all__ = [
    "ENTRIES",
    "ENTRY",
    "FIELD",
    "FIELDS",
    "PAIR",
    "PARTS",
    "PER_UNIT",
    "PythonPrinter",
    "TRIPS",
]
