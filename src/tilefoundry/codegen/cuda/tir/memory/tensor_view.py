"""Emitter for ``tir.memory.TensorView`` — emits ``cute::make_tensor`` (plain)
or ``tilefoundry::make_shard_tensor`` (shard) depending on ``layout`` type.
"""

from __future__ import annotations

from functools import reduce
from operator import mul

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.core import Constant
from tilefoundry.ir.tir.memory.tensor_view import TensorView
from tilefoundry.ir.tir.stmts import LetStmt
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shape_helpers import upper_bound
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Dynamic,
    Partial,
    Split,
    shard_layout_local_shape,
)
from tilefoundry.ir.types.shard.shard_layout import ShardLayout as SL


def _total(shape) -> int:
    if not shape:
        return 1
    # Dynamic dims size to envelope upper bound; runtime extent is
    # plumbed via separate ``<param>_shape_<axis>`` scalars.
    return reduce(mul, (upper_bound(s) for s in shape), 1)


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
        scope = _scope_for(topo.name)
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
    scope = _scope_for(topo.name)
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


_TOPOLOGY_SCOPE = {
    "cta": "tilefoundry::TopologyScope::cta",
    "warp": "tilefoundry::TopologyScope::warp",
    "thread": "tilefoundry::TopologyScope::thread",
}


def _scope_for(name: str) -> str:
    # Loud on an unknown level rather than silently defaulting to cta.
    try:
        return _TOPOLOGY_SCOPE[name]
    except KeyError:
        raise ValueError(
            f"unknown topology level {name!r}; expected one of "
            f"{sorted(_TOPOLOGY_SCOPE)}"
        ) from None


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

    scope = _scope_for(topo.name)
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
    emitted directly (``make_coord(1)``). A rank-0 scalar offset is a native
    integer (a kernel-param scalar lowers to an ``int`` argument, a loop
    induction variable is already native), so it is used by name. The legacy
    all-1 offset tensor (``(1,)``) arrives as a kernel-param cute tensor, so its
    single element is read out (``off_tensor(0)``). This is the only
    tensor→scalar case — it is not a general mechanism."""
    if isinstance(index_var, Constant):
        return str(int(index_var.value))
    name = ctx.name_for(index_var)
    shape = getattr(getattr(index_var, "type", None), "shape", ()) or ()
    dims = tuple(getattr(d, "value", d) for d in shape)
    if dims == (1,) and ctx.is_kernel_param(index_var):
        return f"{name}_tensor(0)"
    return name


@register_codegen_cuda(TensorView)
def _emit(let: LetStmt, ctx: CodegenContext) -> None:
    call = let.value
    memory_var = call.args[0]
    var_name = ctx.name_for(let.var)
    layout = call.target.layout

    # Slice view (second arg is the index Var)
    if len(call.args) > 1:
        mem_name = ctx.name_for(memory_var)
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
            global_total = _total(memory_var.type.shape)
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
            target_total = _total(let.var.type.shape)
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
