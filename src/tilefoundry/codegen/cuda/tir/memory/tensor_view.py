"""Emitter for ``tir.memory.TensorView`` — emits ``cute::make_tensor`` (plain)
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
from tilefoundry.ir.core import Constant
from tilefoundry.ir.tir.memory.tensor_view import TensorView
from tilefoundry.ir.tir.stmts import LetStmt
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shape_helpers import shape_numel_upper_bound, upper_bound
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Dynamic,
    Partial,
    Split,
    shard_layout_local_shape,
)
from tilefoundry.ir.types.shard.shard_layout import ShardLayout as SL


def _render_layout(shape, strides) -> str:
    """cute::Layout<cute::Shape<Int<N>...>, cute::Stride<Int<M>...>>"""
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
        # try exact id match
        entry = ctx._mesh_aliases.get(id(mesh))
        if entry:
            return entry[0]  # alias name
        # try structural fallback: compare inline type string
        topo = mesh.topology
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
    topo = mesh.topology
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
    """Full ShardLayout<...> C++ type string.

    """
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
    """Build the per-axis layout and mesh as runtime C++ *values* and return
    ``(preamble_lines, value_expr)`` for constructing a ``ShardLayout`` value.

    For an all-static layout the constructed value's ``decltype`` is the type
    ``_render_shard_layout_type`` would emit, so a ShardTensor built from it
    carries its layout as a stored value (read back by ``local``) without
    changing the type. Static dims emit ``cute::Int<N>{}``.

    With *dim_var_runtime* (a ``DimVar`` name → kernel shape-scalar map) a
    dynamic global dim emits its runtime scalar, and a launch-provided
    (``None``) CTA mesh extent emits ``program_dim<cta>()``. Without the map
    any dynamic extent raises, so a static call site can never silently take
    the dynamic path; an unmapped dynamic dim also raises rather than falling
    back to an envelope bound.
    """
    sll, ml, topo = sl.layout, sl.mesh.layout, sl.mesh.topology

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

    # A single ``None`` mesh axis is the launch-provided (dynamic) CTA extent →
    # ``program_dim<cta>()``; only a 'cta' topology may carry it and at most
    # one axis may be dynamic.
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
    # A launch-provided CTA extent has no compile-time size; the value carries
    # the real extent, so the Topology type parameter is an inert placeholder.
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


def _coord_ref(index_var, ctx: CodegenContext) -> str:
    """Render a ``local_tile`` coordinate. A compile-time integer literal is
    emitted directly (``make_coord(1)``). A rank-0 scalar is a native integer (a
    kernel-param scalar lowers to an ``int`` argument, a loop induction variable
    is already native), so it is used by name. A one-element ``(1,)`` offset
    tensor (a ``cache_update`` ``cur_pos`` / gather index) is a cute tensor whose
    single element is read out (``off_tensor(0)`` for a kernel param, ``off(0)``
    otherwise). Any other rank fails closed — there is no general
    tensor→coordinate mechanism."""
    if isinstance(index_var, Constant):
        return str(int(index_var.value))
    name = ctx.name_for(index_var)
    shape = getattr(getattr(index_var, "type", None), "shape", ()) or ()
    dims = tuple(getattr(d, "value", d) for d in shape)
    if dims == ():
        return name
    if dims == (1,):
        return f"{name}_tensor(0)" if ctx.is_kernel_param(index_var) else f"{name}(0)"
    raise NotImplementedError(
        f"local_tile coordinate from a rank-{len(dims)} offset {dims} "
        "is not supported"
    )


@register_codegen_cuda(TensorView)
def _emit(let: LetStmt, ctx: CodegenContext) -> None:
    call = let.value
    memory_var = call.args[0]
    var_name = ctx.name_for(let.var)
    layout = call.target.layout

    # Slice view: one coordinate per axis after the memory source. A single
    # coordinate selects a flat window over the coalesced per-thread view
    # (cache_update / 1-D insert_slice); multiple coordinates select a per-axis
    # N-D window (rank-N insert_slice). The coordinate count is validated below
    # against the layout-derived rank — a sharded axis adds a per-thread cute
    # axis, so it is not the logical rank.
    if len(call.args) > 1:
        mem_name = ctx.name_for(memory_var)
        # rank-N window (an offset per axis): a true N-D ``local_tile`` over the
        # per-axis view, so every coordinate and the per-axis window shape reach
        # codegen (no flat collapse). A single coordinate keeps the flat path.
        if len(call.args) > 2:
            logical_coords = call.args[1:]
            dst_layout = getattr(memory_var.type, "layout", None)
            win_layout = getattr(let.var.type, "layout", None)
            if (
                not isinstance(dst_layout, SL)
                or not isinstance(win_layout, SL)
                or ctx.is_kernel_param(memory_var)
            ):
                raise NotImplementedError(
                    "rank-N insert_slice window is only supported over a "
                    "locally allocated sharded (ShardTensor) destination"
                )
            # A reshard adds a per-thread cute axis for each sharded (``Split``)
            # axis, and the local buffer alloc drops every extent-1 axis. Tile
            # over ``local(...)`` (the per-thread buffer) in that same reduced
            # space: the logical offsets bind to the non-split axes in order,
            # and only the retained (extent > 1) axes reach the tile — a dropped
            # axis is degenerate (single position), so its offset is 0.
            dst_local = shard_layout_local_shape(dst_layout)
            win_local = shard_layout_local_shape(win_layout)
            split_axes = {a.axis for a in dst_layout.attrs if isinstance(a, Split)}
            non_split = [a for a in range(len(dst_local)) if a not in split_axes]
            if len(logical_coords) != len(non_split):
                raise ValueError(
                    f"insert_slice: {len(logical_coords)} offsets for a rank-"
                    f"{len(non_split)} destination"
                )
            coord_of = dict(zip(non_split, logical_coords))
            kept = [a for a in range(len(dst_local)) if int(upper_bound(dst_local[a])) != 1]
            if any(a in split_axes for a in kept):
                raise NotImplementedError(
                    "rank-N insert_slice with a per-thread extent > 1 on a "
                    "sharded axis is not supported"
                )
            shape_args = ", ".join(
                f"cute::Int<{int(upper_bound(win_local[a]))}>{{}}" for a in kept
            )
            coord_args = ", ".join(_coord_ref(coord_of[a], ctx) for a in kept)
            ctx.emit(
                f"auto {var_name} = cute::local_tile("
                f"tilefoundry::local({mem_name}), "
                f"cute::make_shape({shape_args}), "
                f"cute::make_coord({coord_args}));"
            )
            return
        index_var = call.args[1]
        if isinstance(getattr(memory_var.type, "layout", None), SL):
            # A sharded intermediate buffer is a ShardTensor; project to the
            # per-thread cute tensor before ``local_tile`` (an in-place
            # ``insert_slice`` / ``cache_update`` window over a carried buffer).
            # ``local()`` coalesces the per-shard layout to a flat 1-D view, so
            # the tile size is the window's *total per-shard element count* and
            # the coord indexes that flat view in whole-window blocks.
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
            # The tensor is a flat rank-1 cute view (a kernel param is wrapped as
            # a 1-D ``Shape<total>`` tensor), so the tile is the window's total
            # element count and the coord indexes it in whole-window blocks.
            K = reduce(
                mul, (int(upper_bound(s)) for s in let.var.type.shape), 1
            )
        ctx.emit(
            f"auto {var_name} = cute::local_tile("
            f"{tensor_ref}, "
            f"cute::make_shape(cute::Int<{K}>{{}}), "
            f"cute::make_coord({_coord_ref(index_var, ctx)}));"
        )
        return

    if isinstance(layout, SL):
        mem_name = ctx.name_for(memory_var)

        if ctx.is_kernel_param(memory_var):
            # Kernel param: memory is a Cute gmem tensor, wrap directly.
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
            # PtrOf result: memory is a raw pointer.  Construct a
            # per-thread cute tensor then wrap as ShardTensor.
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
