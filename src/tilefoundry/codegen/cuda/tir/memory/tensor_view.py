"""Emitter for ``tir.memory.TensorView``.

Emitter for ``tir.memory.TensorView`` — emits ``cute::make_tensor`` (plain)
or ``tilefoundry::make_shard_tensor`` (shard) depending on ``layout`` type.
"""

from __future__ import annotations

from functools import reduce
from operator import mul

from tilefoundry.codegen.cuda.context import (
    CodegenContext,
    register_codegen_cuda,
    topology_scope_str,
)
from tilefoundry.ir.core import Call, Constant
from tilefoundry.ir.tir.memory.tensor_view import TensorView
from tilefoundry.ir.tir.stmts import LetStmt
from tilefoundry.ir.types.dim import DimAdd, DimMul, DimSub, DimVar
from tilefoundry.ir.types.shape_helpers import shape_numel_upper_bound, upper_bound
from tilefoundry.ir.types.shard import c_order_strides
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Dynamic,
    Partial,
    Split,
    shard_layout_local_shape,
)
from tilefoundry.ir.types.shard.shard_layout import ShardLayout as SL
from tilefoundry.ir.visitor import ExprVisitor


def _render_layout(shape, strides) -> str:
    """Render a CuTe layout type."""
    shape_args = ", ".join(f"cute::Int<{s}>" for s in shape)
    stride_args = ", ".join(f"cute::Int<{s}>" for s in strides)
    return (
        f"cute::Layout<"
        f"cute::Shape<{shape_args}>, "
        f"cute::Stride<{stride_args}>>"
    )


def _render_mesh_type(mesh, ctx=None) -> str:
    """tilefoundry::Mesh<...> — uses scope alias if registered in ctx."""
    if ctx and hasattr(ctx, '_mesh_aliases'):

        entry = ctx._mesh_aliases.get(id(mesh))
        if entry:
            return entry[0]

        topo = mesh.topologies[0]
        scope = topology_scope_str(topo.name)
        ml = mesh.layout
        shape_args = ", ".join(f"cute::Int<{s}>" for s in ml.shape)
        stride_args = ", ".join(f"cute::Int<{s}>" for s in ml.strides)
        inline = (
            f"tilefoundry::Mesh<"
            f"tilefoundry::Topology<{scope}, {topo.size}>, "
            f"cute::Layout<cute::Shape<{shape_args}>, cute::Stride<{stride_args}>>>"
        )
        for alias_name, type_str in ctx._mesh_aliases.values():
            if type_str == inline:
                return alias_name
    topo = mesh.topologies[0]
    scope = topology_scope_str(topo.name)
    ml = mesh.layout
    shape_args = ", ".join(f"cute::Int<{s}>" for s in ml.shape)
    stride_args = ", ".join(f"cute::Int<{s}>" for s in ml.strides)
    return (
        f"tilefoundry::Mesh<"
        f"tilefoundry::Topology<{scope}, {topo.size}>, "
        f"cute::Layout<cute::Shape<{shape_args}>, cute::Stride<{stride_args}>>>"
    )


def _render_attr(a) -> str:
    """Single ShardAttr to C++ type string."""
    if isinstance(a, Split):
        return f"tilefoundry::shard::S<{a.axis}>"
    if isinstance(a, Broadcast):
        return "tilefoundry::shard::B"
    if isinstance(a, Partial):
        return "tilefoundry::shard::P<void>"
    if isinstance(a, Dynamic):
        return "tilefoundry::shard::Dynamic"
    return f"/* unknown attr {type(a).__name__} */"


def _render_shard_layout_type(sl: SL, ctx=None) -> str:
    """Render a full ShardLayout C++ type string."""
    layout_str = _render_layout(sl.layout.shape, sl.layout.strides)
    attrs_str = ", ".join(_render_attr(a) for a in sl.attrs)
    mesh_str = _render_mesh_type(sl.mesh, ctx)
    return (
        f"tilefoundry::ShardLayout<"
        f"{layout_str}, "
        f"cute::tuple<{attrs_str}>, "
        f"{mesh_str}>"
    )




def render_shard_layout_value(var_name: str, sl: SL, dim_var_runtime=None):
    """Render a shard layout as runtime C++ preamble and value expression.

    Static values retain the type produced by the type renderer. Runtime
    dimension mappings supply dynamic globals and ``program_dim<cta>()``
    supplies a launch-provided mesh extent. Missing or unmapped dynamic values
    raise instead of falling back to an envelope bound.
    """
    sll, ml, topo = sl.layout, sl.mesh.layout, sl.mesh.topologies[0]

    def _static_dim(value, what):
        if not isinstance(value, int):
            raise NotImplementedError(
                f"render_shard_layout_value: dynamic {what} ({value!r}) is not "
                f"supported"
            )
        return f"cute::Int<{value}>{{}}"

    def _global_dim(d):
        if isinstance(d, int):
            return f"cute::Int<{d}>{{}}"
        if isinstance(d, DimVar):
            if not dim_var_runtime:
                raise NotImplementedError(
                    f"render_shard_layout_value: dynamic layout dim {d.name!r} "
                    f"requires a runtime shape mapping"
                )
            scalar = dim_var_runtime.get(d.name)
            if scalar is None:
                raise ValueError(
                    f"render_shard_layout_value: dynamic layout dim {d.name!r} "
                    f"has no runtime shape scalar"
                )
            return scalar
        raise NotImplementedError(
            f"render_shard_layout_value: unsupported layout dim {d!r}"
        )




    n_dynamic = sum(1 for d in ml.shape if d is None)
    if n_dynamic > 1:
        raise NotImplementedError(
            "render_shard_layout_value: at most one dynamic (launch-provided) "
            "CTA mesh axis is supported"
        )
    if n_dynamic == 1 and topo.name != "cta":
        raise NotImplementedError(
            f"render_shard_layout_value: a dynamic (None) mesh extent is only "
            f"valid on a 'cta' topology, got {topo.name!r}"
        )
    if n_dynamic == 1 and not dim_var_runtime:
        raise NotImplementedError(
            "render_shard_layout_value: a dynamic CTA mesh extent requires a "
            "runtime shape mapping"
        )

    def _mesh_dim(d):
        if d is None:
            return "tilefoundry::program_dim<tilefoundry::TopologyScope::cta>()"
        return _static_dim(d, "mesh layout dim")

    sl_var = f"{var_name}__sl_layout"
    ml_var = f"{var_name}__mesh_layout"
    mesh_var = f"{var_name}__mesh"

    sl_shape = ", ".join(_global_dim(d) for d in sll.shape)
    sl_stride = ", ".join(_static_dim(s, "shard layout stride") for s in sll.strides)
    ml_shape = ", ".join(_mesh_dim(d) for d in ml.shape)
    ml_stride = ", ".join(_static_dim(s, "mesh layout stride") for s in ml.strides)

    scope = topology_scope_str(topo.name)


    topo_size = topo.size if isinstance(topo.size, int) else 0
    attrs = ", ".join(_render_attr(a) for a in sl.attrs)
    preamble = [
        f"auto {sl_var} = cute::make_layout("
        f"cute::make_shape({sl_shape}), cute::make_stride({sl_stride}));",
        f"auto {ml_var} = cute::make_layout("
        f"cute::make_shape({ml_shape}), cute::make_stride({ml_stride}));",
        f"tilefoundry::Mesh<tilefoundry::Topology<{scope}, {topo_size}>, "
        f"decltype({ml_var})> {mesh_var}{{{ml_var}}};",
    ]
    value_expr = (
        f"tilefoundry::ShardLayout<decltype({sl_var}), cute::tuple<{attrs}>, "
        f"decltype({mesh_var})>{{{sl_var}, {mesh_var}}}"
    )
    return preamble, value_expr


_COORD_OPERATORS = {DimAdd: "+", DimSub: "-", DimMul: "*"}


class _CoordinateVisitor(ExprVisitor[str]):
    def __init__(self, ctx: CodegenContext) -> None:
        super().__init__()
        self.ctx = ctx

    def visit_Constant(self, expr: Constant) -> str:
        return str(int(expr.value))

    def visit_Call(self, expr: Call) -> str:
        operator = _COORD_OPERATORS.get(type(expr.target))
        if operator is not None:
            lhs, rhs = (self.visit(arg) for arg in expr.args)
            return f"({lhs} {operator} {rhs})"
        return self._leaf(expr)

    def _leaf(self, expr) -> str:
        name = self.ctx.name_for(expr)
        shape = getattr(getattr(expr, "type", None), "shape", ()) or ()
        dims = tuple(getattr(d, "value", d) for d in shape)
        if dims == ():
            return name
        if dims == (1,):
            return f"{name}_tensor(0)" if self.ctx.is_kernel_param(expr) else f"{name}(0)"
        raise NotImplementedError(
            f"local_tile coordinate from a rank-{len(dims)} offset {dims} "
            "is not supported"
        )

    def default_visit(self, expr) -> str:
        return self._leaf(expr)


def _coord_ref(index_var, ctx: CodegenContext) -> str:
    """Render a compile-time, scalar, or one-element absolute coordinate.

    Integer literals become static coordinates; rank-zero scalars use their
    native names; one-element offset tensors read element zero. Dim arithmetic
    renders as the arithmetic itself: a multiplication preserves grid output
    placement after an ordinal is converted to an element start, and an addition
    moves a window's base by a compile-time offset. Other forms fail closed.
    """
    return _CoordinateVisitor(ctx).visit(index_var)


@register_codegen_cuda(TensorView)
def _emit(let: LetStmt, ctx: CodegenContext) -> None:
    call = let.value
    memory_var = call.args[0]
    var_name = ctx.name_for(let.var)
    layout = call.target.layout







    if len(call.args) > 1:
        mem_name = ctx.name_for(memory_var)



        if len(call.args) > 2:
            logical_coords = call.args[1:]
            dst_layout = getattr(memory_var.type, "layout", None)
            if isinstance(dst_layout, SL):
                tensor_ref = f"tilefoundry::local({mem_name})"
                dst_local = shard_layout_local_shape(dst_layout)
                split_axes = {a.axis for a in dst_layout.attrs if isinstance(a, Split)}
                non_split = [a for a in range(len(dst_local)) if a not in split_axes]
                if len(logical_coords) != len(non_split):
                    raise ValueError(
                        f"tensor_view: {len(logical_coords)} offsets for "
                        f"{len(non_split)} local axes"
                    )
                entries = tuple(zip(non_split, logical_coords, let.var.type.shape))
                kept = tuple(
                    entry for entry in entries if int(upper_bound(dst_local[entry[0]])) != 1
                )
                shape = tuple(window_dim for _, _, window_dim in kept)
                coords = tuple(coord for _, coord, _ in kept)
            else:
                tensor_ref = f"{mem_name}_tensor" if ctx.is_kernel_param(memory_var) else mem_name
                source_shape = tuple(memory_var.type.shape)
                source_shape_args = ", ".join(
                    f"cute::Int<{int(upper_bound(dim))}>{{}}" for dim in source_shape
                )
                source_stride_args = ", ".join(
                    f"cute::Int<{stride}>{{}}"
                    for stride in c_order_strides(
                        tuple(int(upper_bound(dim)) for dim in source_shape)
                    )
                )
                source_name = f"{var_name}__source"
                ctx.emit(
                    f"auto {source_name} = cute::make_tensor("
                    f"{tensor_ref}.data(), cute::make_layout("
                    f"cute::make_shape({source_shape_args}), "
                    f"cute::make_stride({source_stride_args})));"
                )
                tensor_ref = source_name
                shape = tuple(let.var.type.shape)
                if len(logical_coords) != len(shape):
                    raise ValueError(
                        f"tensor_view: {len(logical_coords)} offsets for rank-{len(shape)} view"
                    )
                coords = tuple(logical_coords)
            shape_args = ", ".join(f"cute::Int<{int(upper_bound(dim))}>{{}}" for dim in shape)
            coord_args = ", ".join(_coord_ref(coord, ctx) for coord in coords)
            zero_args = ", ".join("0" for _ in coords)
            offset_name = f"{var_name}__offset"
            ctx.emit(
                f"auto {offset_name} = cute::domain_offset("
                f"cute::make_coord({coord_args}), {tensor_ref});"
            )
            ctx.emit(
                f"auto {var_name} = cute::local_tile("
                f"{offset_name}, "
                f"cute::make_shape({shape_args}), "
                f"cute::make_coord({zero_args}));"
            )
            return
        index_var = call.args[1]
        if isinstance(getattr(memory_var.type, "layout", None), SL):






            tensor_ref = f"tilefoundry::local({mem_name})"
            win_layout = getattr(let.var.type, "layout", None)
            if isinstance(win_layout, SL):
                local_shape = shard_layout_local_shape(win_layout)
            else:
                local_shape = tuple(let.var.type.shape)
            K = reduce(mul, (int(upper_bound(s)) for s in local_shape), 1)
        else:
            if ctx.is_kernel_param(memory_var):
                tensor_ref = f"{mem_name}_tensor"
            else:
                tensor_ref = mem_name



            K = reduce(
                mul, (int(upper_bound(s)) for s in let.var.type.shape), 1
            )
        offset_name = f"{var_name}__offset"
        ctx.emit(
            f"auto {offset_name} = cute::domain_offset({_coord_ref(index_var, ctx)}, {tensor_ref});"
        )
        ctx.emit(
            f"auto {var_name} = cute::local_tile("
            f"{offset_name}, "
            f"cute::make_shape(cute::Int<{K}>{{}}), "
            f"cute::make_coord(0));"
        )
        return

    if isinstance(layout, SL):
        mem_name = ctx.name_for(memory_var)

        if ctx.is_kernel_param(memory_var):

            tensor_ref = f"{mem_name}_tensor"
            global_total = shape_numel_upper_bound(memory_var.type.shape)
            global_layout = (
                f"cute::make_layout(cute::Shape<cute::Int<{global_total}>>{{}})"
            )
            preamble, shard_value = render_shard_layout_value(
                var_name, layout, getattr(ctx, "_dim_var_runtime", None)
            )
            for line in preamble:
                ctx.emit(line)
            ctx.emit(
                f"auto {var_name} = tilefoundry::make_shard_tensor("
                f"{tensor_ref}, {global_layout}, {shard_value});"
            )
        else:


            local_shape = shard_layout_local_shape(layout)
            local_shape = tuple(s for s in local_shape if s != 1) or (1,)
            if len(local_shape) > 1:
                shape_args = ", ".join(f"cute::Int<{int(s)}>" for s in local_shape)
                tensor_layout = f"cute::make_layout(cute::Shape<{shape_args}>{{}})"
            else:
                tensor_layout = f"cute::make_layout(cute::Shape<cute::Int<{int(local_shape[0])}>>{{}})"
            ctx.emit(
                f"auto {var_name}_tensor = cute::make_tensor("
                f"{mem_name}, {tensor_layout});"
            )
            target_total = shape_numel_upper_bound(let.var.type.shape)
            target_global = (
                f"cute::make_layout(cute::Shape<cute::Int<{target_total}>>{{}})"
            )
            preamble, shard_value = render_shard_layout_value(
                var_name, layout, getattr(ctx, "_dim_var_runtime", None)
            )
            for line in preamble:
                ctx.emit(line)
            ctx.emit(
                f"auto {var_name} = tilefoundry::make_shard_tensor("
                f"{var_name}_tensor, {target_global}, {shard_value});"
            )
