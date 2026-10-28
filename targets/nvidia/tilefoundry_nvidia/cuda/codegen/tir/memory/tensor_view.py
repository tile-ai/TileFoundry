"""Emitter for ``tir.memory.TensorView``.

Emitter for ``tir.memory.TensorView`` — emits ``cute::make_tensor`` (plain)
or ``tilefoundry::make_shard_tensor`` (shard) depending on ``layout`` type.
"""

from __future__ import annotations

from dataclasses import replace
from functools import reduce
from operator import mul

from tilefoundry.ir.core import Call, Constant, Tuple, Var
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.tir.memory.ptr_of import PtrOf
from tilefoundry.ir.tir.memory.tensor_view import TensorView, _byte_offset
from tilefoundry.ir.tir.stmts import LetStmt
from tilefoundry.ir.tir.sync import participation
from tilefoundry.ir.types import ComposedLayout
from tilefoundry.ir.types.dim import DimAdd, DimFloorDiv, DimMod, DimMul, DimSub, DimVar
from tilefoundry.ir.types.layout import Layout, LayoutBase, flatten
from tilefoundry.ir.types.mesh import flattened_levels
from tilefoundry.ir.types.shard_layout import (
    Broadcast,
    Dynamic,
    Partial,
    Split,
    register_strides,
    shard_layout_local_shape,
)
from tilefoundry.ir.types.shard_layout import ShardLayout as SL
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.types.swizzle_layout import get_swizzle_portion
from tilefoundry.ir.types.utils import shape_numel_upper_bound, static_dim_value, upper_bound
from tilefoundry.ir.visitor import ExprVisitor
from tilefoundry.visitor_registry.registries import Role, register_codegen
from tilefoundry_nvidia.cuda import CudaTarget
from tilefoundry_nvidia.cuda.codegen.context import (
    CudaCodegenContext,
    scalar_read,
    topology_scope_str,
)
from tilefoundry_nvidia.cuda.codegen.tir.cute_layout import render_layout_type, render_layout_value
from tilefoundry_nvidia.cuda.codegen.tir.reduce import REDUCE_TAG
from tilefoundry_nvidia.cuda.codegen.tir.stmts.mesh_scope import (
    mesh_type,
    program_topologies,
)
from tilefoundry_nvidia.cuda.codegen.tir.tuple_get_item import constant_index


def _swizzle_args(swizzle) -> str:
    """``B, M, S`` as the ``cute::Swizzle`` template arguments."""
    return f"{swizzle.bits}, {swizzle.base}, {swizzle.shift}"


def _render_layout_type(layout: LayoutBase) -> str:
    """Render a CuTe layout type.

    A swizzled composed layout keeps its shape: CuTe states it as a
    ``ComposedLayout`` of the swizzle, a static offset and the layout
    underneath. The XOR mapping is not expressible as strides, so there is no
    affine form to fall back to. CuTe encodes the single rank-0 address as a
    singleton mode.
    """
    swizzle = get_swizzle_portion(layout)
    if swizzle is not None:
        return (
            f"cute::ComposedLayout<cute::Swizzle<{_swizzle_args(swizzle)}>, "
            f"cute::Int<{int(layout.offset)}>, {_render_layout_type(layout.outer)}>"
        )
    if isinstance(layout, Layout):
        if not layout.shape:
            return "cute::Layout<cute::Shape<cute::Int<1>>, cute::Stride<cute::Int<0>>>"
        return render_layout_type(layout.shape, layout.strides)
    raise NotImplementedError(f"tensor_view: no CuTe layout type for {type(layout).__name__}")


def _render_layout_value(layout: LayoutBase, dim, stride) -> str:
    """Render a CuTe layout value, the mirror of :func:`_render_layout_type`.

    *dim* and *stride* render one shape entry and one stride entry, which is
    where a runtime-provided extent reaches the emitted layout. CuTe encodes
    the single rank-0 address as a singleton mode.
    """
    swizzle = get_swizzle_portion(layout)
    if swizzle is not None:
        return (
            f"cute::make_composed_layout("
            f"cute::Swizzle<{_swizzle_args(swizzle)}>{{}}, "
            f"cute::Int<{int(layout.offset)}>{{}}, "
            f"{_render_layout_value(layout.outer, dim, stride)})"
        )
    if isinstance(layout, Layout):
        if not layout.shape:
            return "cute::make_layout(cute::Shape<cute::Int<1>>{}, cute::Stride<cute::Int<0>>{})"
        return render_layout_value(layout.shape, layout.strides, dim, stride)
    raise NotImplementedError(f"tensor_view: no CuTe layout value for {type(layout).__name__}")


def _scope_mesh_value(mesh, ctx) -> "str | None":
    """The name of the enclosing scope's mesh object, when it is this mesh."""
    if ctx is None or not hasattr(ctx, "_mesh_aliases"):
        return None
    entry = ctx._mesh_aliases.get(id(mesh))
    if entry is None:
        inline = mesh_type(mesh)
        entry = next((e for e in ctx._mesh_aliases.values() if e[1] == inline), None)
    if entry is None:
        return None
    alias = entry[0]
    return alias[: -len("_mesh_t")] + "_mesh" if alias.endswith("_mesh_t") else None


@register_codegen(CudaTarget, Role.EMIT, MeshCoord)
def _mesh_coord_value(call: Call, ctx) -> str:
    """This unit's coordinate along one static axis of its enclosing mesh."""
    axis = static_dim_value(call.args[0])
    if axis is None:
        raise NotImplementedError("CUDA MeshCoord requires a static axis")
    mesh = _scope_mesh_value(call.target.mesh, ctx)
    if mesh is None:
        raise ValueError("CUDA MeshCoord requires its enclosing mesh binding")
    return f"cute::get<{axis}>(cute::flatten(tilefoundry::mesh_coords({mesh}, tilefoundry::program_ids())))"


def _render_mesh_type(mesh, ctx=None) -> str:
    """tilefoundry::Mesh<...> — uses scope alias if registered in ctx."""
    inline = mesh_type(mesh)
    if ctx and hasattr(ctx, "_mesh_aliases"):
        entry = ctx._mesh_aliases.get(id(mesh))
        if entry:
            return entry[0]
        for alias_name, type_str in ctx._mesh_aliases.values():
            if type_str == inline:
                return alias_name
    return inline


def _partial_reduction_tag(reduction: str) -> str:
    """The C++ reduce tag a ``Partial``'s reduction names.

    See [shard §6](docs/spec/shard.md#6-shardattr).
    """
    try:
        kind = ReduceKind(reduction)
    except ValueError as error:
        raise NotImplementedError(
            f"tensor_view: Partial reduction {reduction!r} names no ReduceKind, "
            f"so there is no C++ reduce tag for it; expected one of "
            f"{sorted(k.value for k in ReduceKind)}"
        ) from error
    return REDUCE_TAG[kind]


def _render_attr(a) -> str:
    """Single ShardAttr to C++ type string."""
    if isinstance(a, Split):
        return f"tilefoundry::shard::S<{a.axis}>"
    if isinstance(a, Broadcast):
        return "tilefoundry::shard::B"
    if isinstance(a, Partial):
        return f"tilefoundry::shard::P<{_partial_reduction_tag(a.reduction)}>"
    if isinstance(a, Dynamic):
        return "tilefoundry::shard::Dynamic"
    return f"/* unknown attr {type(a).__name__} */"


def _render_shard_layout_type(sl: SL, ctx=None) -> str:
    """Render a full ShardLayout C++ type string."""
    layout_str = _render_layout_type(sl.layout)
    attrs_str = ", ".join(_render_attr(a) for a in sl.attrs)
    mesh_str = _render_mesh_type(sl.mesh, ctx)
    return f"tilefoundry::ShardLayout<{layout_str}, cute::tuple<{attrs_str}>, {mesh_str}>"


def _composed_mesh_layout(positions: str, base: int) -> str:
    """A mesh layout value expression, its slice origin folded in.

    [runtime §2.3.2](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/runtime.md#232-layoutmeshcuh).
    """
    if not base:
        return positions
    return f"cute::make_composed_layout(cute::identity{{}}, cute::Int<{base}>{{}}, {positions})"


def shard_engine_layout(shard: SL, storage: StorageKind) -> LayoutBase:
    """Read the physical layout of the storage this shard tensor wraps.

    Global memory and CTA-shared storage retain their declared address steps.
    Registers use the same local extents and numbering as the emitted shard.
    """
    physical = flatten(shard.layout)
    if storage is StorageKind.RMEM:
        return Layout(shard_layout_local_shape(shard), register_strides(shard))
    if storage is StorageKind.GMEM or (
        storage is StorageKind.SMEM and program_topologies(shard.mesh)[0].name == "thread"
    ):
        return physical
    local = shard_layout_local_shape(shard)
    if isinstance(physical, Layout):
        return replace(physical, shape=local)
    return replace(physical, outer=replace(physical.outer, shape=local))


def render_shard_layout_value(var_name: str, sl: SL, dynamic_extents=None, storage=None, ctx=None):
    """Render a shard layout as runtime C++ preamble and value expression.

    Static values retain the type produced by the type renderer. Runtime
    dimension mappings supply dynamic globals and ``program_dim<cta>()``
    supplies a launch-provided mesh extent. Missing or unmapped dynamic values
    raise instead of falling back to an envelope bound. A sliced mesh reaches
    the value the same way it reaches the type: through the mesh layout directly.
    *storage* is the engine's storage class: ``rmem`` takes
    ``register_strides``, anything else the layout's own.
    """
    sll = sl.layout
    if storage is StorageKind.RMEM:
        if get_swizzle_portion(sll) is not None:
            raise NotImplementedError(
                "render_shard_layout_value: a Swizzle states how a shared-memory "
                "bank pattern is arranged; a register engine has no such addresses, "
                "and rebuilding this layout from register strides would drop it"
            )
        sll = Layout(shape=sll.shape, strides=register_strides(sl))
    mesh_base = 0
    if isinstance(sl.mesh.layout, ComposedLayout):
        if sl.mesh.layout.outer is None:
            raise NotImplementedError(
                "render_shard_layout_value: a sliced mesh needs its participating box "
                "as a strided Layout; this states an identity box, with no sub-box"
            )
        mesh_base = sl.mesh.layout.offset
        participation(sl.mesh)
    stated = sl.mesh.layout
    if isinstance(stated, ComposedLayout):
        stated = stated.outer
    topos = program_topologies(sl.mesh)
    if len(topos) > 1 and mesh_base:
        raise NotImplementedError(
            "render_shard_layout_value: a mesh naming several levels cannot also "
            "be sliced; the slice and the level boundary would both be deciding "
            "which positions these are"
        )
    ml_base = int(mesh_base)
    mesh_axes = flatten(stated)
    topo = topos[0]

    def _static_dim(value, what):
        if not isinstance(value, int):
            raise NotImplementedError(
                f"render_shard_layout_value: dynamic {what} ({value!r}) is not supported"
            )
        return f"cute::Int<{value}>{{}}"

    def _global_dim(d):
        if isinstance(d, int):
            return f"cute::Int<{d}>{{}}"
        if isinstance(d, DimVar):
            if not dynamic_extents:
                raise NotImplementedError(
                    f"render_shard_layout_value: dynamic layout dim {d.name!r} "
                    f"requires a runtime shape mapping"
                )
            scalar = dynamic_extents.get(d.name)
            if scalar is None:
                raise ValueError(
                    f"render_shard_layout_value: dynamic layout dim {d.name!r} "
                    f"has no runtime shape scalar"
                )
            return scalar
        raise NotImplementedError(f"render_shard_layout_value: unsupported layout dim {d!r}")

    n_dynamic = sum(1 for d in mesh_axes.shape if d is None)
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
    if n_dynamic == 1 and not dynamic_extents:
        raise NotImplementedError(
            "render_shard_layout_value: a dynamic CTA mesh extent requires a runtime shape mapping"
        )

    def _mesh_dim(d):
        if d is None:
            return "tilefoundry::program_dim<tilefoundry::TopologyScope::cta>()"
        return _static_dim(d, "mesh layout dim")

    def _mesh_stride(s):
        return _static_dim(s, "mesh layout stride")

    sl_var = f"{var_name}__sl_layout"
    ml_var = f"{var_name}__mesh_layout"
    mesh_var = f"{var_name}__mesh"

    if len(topos) == 1:
        mesh_shape, mesh_strides = mesh_axes.shape, mesh_axes.strides
    else:
        arrangements = flattened_levels(replace(sl.mesh, layout=stated))
        mesh_shape = tuple(shape for shape, _ in arrangements)
        mesh_strides = tuple(strides for _, strides in arrangements)
    scope = ", ".join(topology_scope_str(t.name) for t in topos)
    positions = render_layout_value(mesh_shape, mesh_strides, _mesh_dim, _mesh_stride)
    mesh_layout = _composed_mesh_layout(positions, ml_base)

    attrs = ", ".join(_render_attr(a) for a in sl.attrs)
    sl_layout = _render_layout_value(
        sll, _global_dim, lambda s: _static_dim(s, "shard layout stride")
    )
    preamble = [f"auto {sl_var} = {sl_layout};"]
    scope_mesh = _scope_mesh_value(sl.mesh, ctx)
    if scope_mesh is not None:
        mesh_var = scope_mesh
    else:
        preamble += [
            f"auto {ml_var} = {mesh_layout};",
            f"tilefoundry::Mesh<decltype({ml_var}), {scope}> {mesh_var}{{{ml_var}}};",
        ]
    value_expr = (
        f"tilefoundry::ShardLayout<decltype({sl_var}), cute::tuple<{attrs}>, "
        f"decltype({mesh_var})>{{{sl_var}, {mesh_var}}}"
    )
    return preamble, value_expr


_COORD_OPERATORS = {DimAdd: "+", DimSub: "-", DimMul: "*", DimFloorDiv: "/", DimMod: "%"}


class _CoordinateVisitor(ExprVisitor[str]):
    def visit_Constant(self, expr: Constant, ctx: CudaCodegenContext) -> str:
        return str(int(expr.value))

    def visit_Call(self, expr: Call, ctx: CudaCodegenContext) -> str:
        operator = _COORD_OPERATORS.get(type(expr.target))
        if operator is not None:
            lhs, rhs = (self.visit(arg, ctx) for arg in expr.args)
            return f"({lhs} {operator} {rhs})"
        if isinstance(expr.target, MeshCoord):
            return _mesh_coord_value(expr, ctx)
        if isinstance(expr.target, TupleGetItem):
            held, index = expr.args
            if isinstance(index, Constant):
                values = ctx.tuple_for(held) if isinstance(held, Var) else held
                return self.visit(values.elements[constant_index(index)], ctx)
            return f"{ctx.name_for(held)}[{self.visit(index, ctx)}]"
        raise NotImplementedError(
            f"local_tile coordinate: Call target {type(expr.target).__name__!r} is not supported"
        )

    def _leaf(self, expr, ctx: CudaCodegenContext) -> str:
        shape = getattr(getattr(expr, "type", None), "shape", ()) or ()
        dims = tuple(getattr(d, "value", d) for d in shape)
        if dims == ():
            return scalar_read(expr, ctx) if getattr(getattr(expr, "type", None), "storage", None) is StorageKind.RMEM else ctx.name_for(expr)
        if dims == (1,):
            return scalar_read(expr, ctx)
        raise NotImplementedError(
            f"local_tile coordinate from a rank-{len(dims)} offset {dims} is not supported"
        )

    def default_visit(self, expr, ctx: CudaCodegenContext) -> str:
        return self._leaf(expr, ctx)


def _coord_ref(index_var, ctx: CudaCodegenContext) -> str:
    """Render a compile-time, scalar, or one-element absolute coordinate.

    Integer literals become static coordinates; scalar values use their native
    names, while scalar register tensors and one-element offsets read element zero. Dim arithmetic
    renders as the arithmetic itself: a multiplication preserves grid output
    placement after an ordinal is converted to an element start, and an addition
    moves a window's base by a compile-time offset. A mesh coordinate reads this
    unit's position along its axis. Other forms fail closed.
    """
    return _CoordinateVisitor().visit(index_var, ctx)


def _tensor_ref(var, ctx: CudaCodegenContext) -> str:
    name = ctx.name_for(var)
    return f"{name}_tensor" if ctx.is_kernel_param(var) else name


def _smem_offset_ref(pointer, dtype, ctx: CudaCodegenContext) -> tuple[str, None]:
    cpp_type = ctx.dtype_to_cpp(dtype)
    return (
        f"cute::make_smem_ptr(reinterpret_cast<{cpp_type} *>({ctx.smem_base()} + {_coord_ref(pointer, ctx)}))",
        None,
    )


def _pointer_ref(pointer, ctx: CudaCodegenContext) -> tuple[str, object | None]:
    """Render a TensorView pointer and return its syntactic source tensor, if any."""
    if isinstance(pointer, Call) and isinstance(pointer.target, PtrOf):
        source = pointer.args[0]
        if isinstance(source, Call) and isinstance(source.target, Slice):
            base = source.args[0]
            layout = base.type.layout
            if not isinstance(layout, Layout):
                raise NotImplementedError("CUDA pointer slices require a plain source layout")
            origin = source.args[1]
            starts = origin.elements if isinstance(origin, Tuple) else (origin,)
            coords = ", ".join(_coord_ref(start, ctx) for start in starts)
            offset = f"{_plain_layout_value(layout)}(cute::make_coord({coords}))"
            pointer, root = _pointer_ref(Call(PtrOf(), (base,), type=pointer.type), ctx)
            return f"({pointer} + {offset})", root
        if isinstance(source, Call) and isinstance(source.target, TensorView):
            return _pointer_ref(source.args[0], ctx)
        if _byte_offset(source):
            return _smem_offset_ref(source, pointer.type.dtype.name, ctx)
        return f"{_tensor_ref(source, ctx)}.data()", source
    return ctx.name_for(pointer), None


def _envelope_dim(value) -> str:
    """A shape entry at its static envelope bound, as a CuTe value."""
    return f"cute::Int<{int(upper_bound(value))}>{{}}"


def _static_value(value) -> str:
    """A static stride entry as a CuTe value."""
    return f"cute::Int<{int(value)}>{{}}"


def _plain_layout_value(layout: LayoutBase) -> str:
    return _render_layout_value(layout, _envelope_dim, _static_value)


@register_codegen(CudaTarget, Role.EMIT, TensorView)
def _emit(let: LetStmt, ctx: CudaCodegenContext) -> None:
    call = let.value
    pointer = call.args[0]
    pointer_ref, memory_var = (
        _smem_offset_ref(pointer, call.type.dtype.name, ctx)
        if _byte_offset(pointer) else _pointer_ref(pointer, ctx)
    )
    var_name = ctx.name_for(let.var)
    layout = call.target.layout

    if len(call.args) > 1:
        if memory_var is None:
            raise NotImplementedError(
                "tensor_view coordinates require a syntactic T.ptr_of(tensor) source"
            )
        mem_name = ctx.name_for(memory_var)

        if len(call.args) > 2:
            logical_coords = call.args[1:]
            dst_layout = getattr(memory_var.type, "layout", None)
            if isinstance(dst_layout, SL):
                tensor_ref = f"tilefoundry::local({mem_name})"
                dst_local = shard_layout_local_shape(dst_layout)
                split_axes = {a.axis for a in dst_layout.attrs if isinstance(a, Split)}
                non_split = [a for a in range(len(dst_local)) if a not in split_axes]
                if len(logical_coords) == len(dst_local):
                    coordinate_axes = range(len(dst_local))
                elif len(logical_coords) == len(non_split):
                    coordinate_axes = non_split
                else:
                    raise ValueError(
                        f"tensor_view: {len(logical_coords)} offsets for "
                        f"{len(non_split)} or {len(dst_local)} local axes"
                    )
                entries = tuple(zip(coordinate_axes, logical_coords, let.var.type.shape))
                kept = tuple(
                    entry for entry in entries if int(upper_bound(dst_local[entry[0]])) != 1
                )
                shape = tuple(window_dim for _, _, window_dim in kept)
                coords = tuple(coord for _, coord, _ in kept)
            else:
                tensor_ref = f"{mem_name}_tensor" if ctx.is_kernel_param(memory_var) else mem_name
                source_shape = tuple(memory_var.type.shape)
                source_strides = Layout(
                    tuple(int(upper_bound(dim)) for dim in source_shape)
                ).strides
                source_name = f"{var_name}__source"
                ctx.emit(
                    f"auto {source_name} = cute::make_tensor("
                    f"{tensor_ref}.data(), "
                    f"{render_layout_value(source_shape, source_strides, _envelope_dim, _static_value)});"
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

            K = reduce(mul, (int(upper_bound(s)) for s in let.var.type.shape), 1)
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
        target_total = shape_numel_upper_bound(let.var.type.shape)
        target_global = f"cute::make_layout(cute::Shape<cute::Int<{target_total}>>{{}})"
        engine_layout = _plain_layout_value(shard_engine_layout(layout, let.var.type.storage))
        ctx.emit(
            f"auto {var_name}_tensor = "
            f"tilefoundry::ops::tensor_view({pointer_ref}, {engine_layout});"
        )
        preamble, shard_value = render_shard_layout_value(
            var_name,
            layout,
            ctx.dynamic_extents,
            getattr(let.var.type, "storage", None),
            ctx,
        )
        for line in preamble:
            ctx.emit(line)
        ctx.emit(
            f"auto {var_name} = tilefoundry::make_shard_tensor("
            f"{var_name}_tensor, {target_global}, {shard_value});"
        )
        return

    ctx.emit(
        f"auto {var_name} = tilefoundry::ops::tensor_view("
        f"{pointer_ref}, {_plain_layout_value(layout)});"
    )
