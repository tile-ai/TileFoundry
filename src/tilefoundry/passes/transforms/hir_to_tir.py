"""Lower HIR functions to explicit-output TIR primitive functions.

Resources become allocation let-bindings and value operations become effect
statements writing destination buffers. Lowering collects a flat item sequence
then folds it into nested TIR. Per-operation handlers dispatch through a registry
and target-owned handlers use only the public lowering context. See
[passes §7.1](docs/spec/passes.md#71-hirtotirpass).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Union

from tilefoundry.ir.core import Call, Constant, Expr, Tuple, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.pattern import DimVarRangePat, locate_dim_var
from tilefoundry.ir.hir.cuda.nn.mma import (
    Mma_SM80_16x8x16 as HirMmaSM80_16x8x16,
)
from tilefoundry.ir.hir.cuda.nn.mma import (
    Wgmma_SM90_64x128x16 as HirWgmma_SM90,
)
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.math.binary import Binary as HirBinary
from tilefoundry.ir.hir.math.clamp import Clamp as HirClamp
from tilefoundry.ir.hir.math.unary import Unary as HirUnary
from tilefoundry.ir.hir.nn.relu import ReLU as HirReLU
from tilefoundry.ir.hir.sharding.reshard import Reshard, _shared_engine_strides
from tilefoundry.ir.hir.tensor.cache_update import CacheUpdate as HirCacheUpdate
from tilefoundry.ir.hir.tensor.cast import Cast as HirCast
from tilefoundry.ir.hir.tensor.full_like import FullLike as HirFullLike
from tilefoundry.ir.hir.tensor.index_select import IndexSelect as HirIndexSelect
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice as HirInsertSlice
from tilefoundry.ir.hir.tensor.reduce import Reduce as HirReduce
from tilefoundry.ir.hir.tensor.reshape import Reshape as HirReshape
from tilefoundry.ir.hir.tensor.slice import Slice as HirSlice
from tilefoundry.ir.hir.tensor.slice import window_base
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem as HirTupleGetItem
from tilefoundry.ir.tir.arith import (
    Binary as TirBinary,
)
from tilefoundry.ir.tir.arith import (
    Unary as TirUnary,
)
from tilefoundry.ir.tir.arith import (
    UnaryKind,
)
from tilefoundry.ir.tir.clamp import Clamp as TirClamp
from tilefoundry.ir.tir.dispatch import DispatchCall
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.memory import AllocTensor as AllocTensorOp
from tilefoundry.ir.tir.memory.copy import Copy
from tilefoundry.ir.tir.memory.fill import Fill
from tilefoundry.ir.tir.memory.ptr_of import PtrOf
from tilefoundry.ir.tir.memory.tensor_view import TensorView
from tilefoundry.ir.tir.nn.relu import ReLU as TirReLU
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.reduce import Reduce as TirReduce
from tilefoundry.ir.tir.shape import ShapeOf, parse_shape_var_name, shape_var_name
from tilefoundry.ir.tir.stmt import Stmt
from tilefoundry.ir.tir.stmts import (
    Abort,
    Evaluate,
    For,
    LetStmt,
    MeshScope,
    Return,
    Sequential,
)
from tilefoundry.ir.tir.symbol_ref import SymbolRef, symbol_call
from tilefoundry.ir.tir.sync import Sync as TirSync
from tilefoundry.ir.types import (
    DType,
    TensorType,
    TupleType,
    callable_type_for_prim_function,
)
from tilefoundry.ir.types.dim import DimMul, is_dim_op_call, simplify_dim
from tilefoundry.ir.types.shape_helpers import shape_numel_upper_bound
from tilefoundry.ir.types.shard import c_order_strides
from tilefoundry.ir.types.shard.layout import Layout as _Layout
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.shard_layout import (
    ShardLayout,
    Split,
    shard_layout_local_shape,
)
from tilefoundry.ir.types.shard.shard_layout import (
    layout_axis_to_tensor_axis as _layout_axis_to_tensor_axis,
)
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.visitor import ExprFunctor, ExprVisitor, ExprWalker
from tilefoundry.passes.pass_base import ModulePass
from tilefoundry.target import CudaTarget, Target, default_target
from tilefoundry.visitor_registry.registries import (
    hir_lowering_registry,
    register_hir_lowering,
)


class _InductionSliceVisitor(ExprVisitor[bool]):
    def __init__(self, induction_var: Var) -> None:
        super().__init__()
        self.induction_var = induction_var

    def visit_Call(self, expr: Call) -> bool:
        if isinstance(expr.target, HirSlice):
            starts = expr.args[1]
            if isinstance(starts, Tuple) and any(
                window_base(start)[0] is self.induction_var for start in starts.elements
            ):
                return True
        target = expr.target
        return any(
            self.visit(child)
            for child in ((*target,) if isinstance(target, HirFunction) else ())
            + expr.args
        )

    def visit_Function(self, expr: HirFunction) -> bool:
        children = () if expr.body is None else (expr.body,)
        children += expr.variants
        children += tuple(converter for _, converter in expr.converters)
        return any(self.visit(child) for child in children)

    def visit_Tuple(self, expr: Tuple) -> bool:
        return any(self.visit(element) for element in expr.elements)

    def visit_GridRegionExpr(self, expr: GridRegionExpr) -> bool:
        return any(
            self.visit(value)
            for value in (*expr.init_args, expr.body, *expr.yield_values, *expr.carried_args)
        )

    def visit_Var(self, expr: Var) -> bool:
        return False

    def visit_Constant(self, expr: Constant) -> bool:
        return False

    def default_visit(self, expr) -> bool:
        return False


def _has_induction_slice(root: Expr, induction_var: Var) -> bool:
    """Whether ``root`` contains a Slice starting at ``induction_var``.

    A window moved by a compile-time offset is still that loop's window, so a
    tail it cannot read is still that loop's tail.
    """
    return _InductionSliceVisitor(induction_var).visit(root)


def _reject_partial_window(region: GridRegionExpr) -> None:
    """Reject a tiled Slice tail until residual lowering has an IR contract."""
    start, extent, step = region.start, region.extent, region.step
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (start, extent, step)
    ):
        return
    if step <= 0 or extent <= start or (extent - start) % step == 0:
        return
    roots = (region.body, *region.yield_values)
    if any(_has_induction_slice(root, region.induction_var) for root in roots):
        raise NotImplementedError(
            "hir_to_tir: non-divisible tile window requires handwritten tail "
            f"lowering (start={start}, extent={extent}, step={step})"
        )


def _eval_call(op, args: tuple) -> Evaluate:
    """Place an effect-ful TIR Op in Stmt position as ``Evaluate(op, args)``."""
    return Evaluate(callable=op, args=args)


def _is_full_layout(layout) -> bool:
    """True when ``layout`` is a full (bijective) embedding.

    True when ``layout`` is a full (bijective) embedding — its
    cosize (``1 + Σ (shape[i]-1)·stride[i]``) equals the product of its
    shape — so the strides describe a complete global gather mapping
    rather than a collapsed per-instance form. Returns ``False`` when the
    strides are unavailable (unmaterialized layout), so callers fall back
    to the shared-engine path instead of crashing.
    """
    strides = getattr(layout, "strides", None)
    if strides is None:
        return False
    shape = layout.shape
    if len(shape) != len(strides):
        return False
    if all(isinstance(d, int) for d in shape) and all(
        isinstance(s, int) for s in strides
    ):
        size = 1
        for d in shape:
            size *= d
        cosize = 1 + sum((shape[i] - 1) * strides[i] for i in range(len(shape)))
        return cosize == size





    if not all(isinstance(d, int) and not isinstance(d, bool) for d in shape[1:]):
        return False
    expected = c_order_strides(tuple(shape))
    return all(
        isinstance(strides[i], int) and strides[i] == expected[i]
        for i in range(len(shape))
    )


def _analyze_cross_warp_workspace(input_ty, reduce_axes):
    """Compute staging requirements for a sharded cross-warp reduction.

    Return workspace size, dtype, and whether a reduced split is intra-warp.
    Zero size means no staging. A cross-warp-only reduction stages each warp,
    lane, and cell separately. Runtime derives the reduction tier and group size
    from operand layouts.
    """
    layout = getattr(input_ty, "layout", None)
    if not isinstance(layout, ShardLayout):
        return 0, input_ty.dtype, True




    layout_shape = tuple(int(s) for s in layout.layout.shape)
    pos_to_axis = _layout_axis_to_tensor_axis(layout_shape, input_ty.shape)
    rank = len(input_ty.shape)
    normalized = tuple(a % rank if a < 0 else a for a in reduce_axes)

    mesh = layout.mesh
    mesh_shape = tuple(mesh.layout.shape)
    topologies = list(mesh.topologies)










    warp_size = 32
    thread_axes = 0
    if topologies:
        last_topo = topologies[-1]
        last_name = last_topo.name if hasattr(last_topo, "name") else ""
        if last_name == "thread":
            prod = 1
            for extent in reversed(mesh_shape):
                ext = int(extent)
                if prod * ext > warp_size:
                    break
                prod *= ext
                thread_axes += 1





    cta_topo_axes: set[int] = set()
    if topologies:
        idx = 0
        for topo in topologies:
            tname = topo.name if hasattr(topo, "name") else ""
            tsize = int(getattr(topo, "size", 1))
            prod = 1
            while idx < len(mesh_shape) and prod < tsize:
                prod *= int(mesh_shape[idx])
                if tname == "cta":
                    cta_topo_axes.add(idx)
                idx += 1

    cross_warp = 1
    group_count = 1
    lane_reduced = False
    for mesh_axis_idx, attr in enumerate(layout.attrs):
        if not isinstance(attr, Split):
            continue
        L = attr.axis
        if not (0 <= L < len(pos_to_axis)):
            continue
        on_reduced = pos_to_axis[L] in normalized

        is_thread = mesh_axis_idx >= len(mesh_shape) - thread_axes
        if is_thread:
            if on_reduced:
                lane_reduced = True
            continue
        if on_reduced and mesh_axis_idx in cta_topo_axes:
            raise NotImplementedError(
                "cross-CTA reduce not yet supported "
                "(reduce_cross_cta placeholder only)"
            )

        mesh_ext = int(mesh_shape[mesh_axis_idx])
        if on_reduced:
            cross_warp *= mesh_ext
        else:
            group_count *= mesh_ext


    total_warps = cross_warp * group_count



    if total_warps <= 1 or cross_warp <= 1:
        return 0, input_ty.dtype, True




    return total_warps, input_ty.dtype, lane_reduced


@dataclass(frozen=True)
class _Bind:
    """Fold marker: introduce a LetStmt binding `var = value`."""
    var: Var
    value: Expr


_Item = Union[_Bind, Stmt]


class _Lowerer:
    """Collect TIR items and expose the per-operation lowering-handler ABI.

    Target-owned handlers use ``lower``, ``fresh``, ``alloc``, ``emit``, and
    ``emit_bind`` only. Core handlers may use private cooperation for dispatch,
    reshard synchronization, and tuple carries. See
    [passes §7.1](docs/spec/passes.md#71-hirtotirpass).
    """
    def __init__(
        self,
        *,
        dispatch_groups: "dict[str, tuple[HirFunction, ...]] | None" = None,
        mangled_registry: "dict[str, PrimFunction] | None" = None,
        caller_fn: "HirFunction | None" = None,
        shape_param_names: "set[tuple[str, int]] | None" = None,
    ) -> None:
        self._cache: dict[int, Var] = {}
        self._items: list[_Item] = []
        self._name_counter = 0


        self._tuple_parts: dict[int, list[Var]] = {}


        self._carry_init: dict[int, Expr] = {}












        self._dispatch_groups = dispatch_groups or {}
        self._mangled_registry = mangled_registry or {}
        self._caller_fn = caller_fn
        self._shape_param_names: set[tuple[str, int]] = (
            shape_param_names if shape_param_names is not None else set()
        )

    def _fresh(self, type_, hint: str = "v") -> Var:
        self._name_counter += 1
        return Var(type=type_, name=f"{hint}{self._name_counter}")

    def _alloc(self, type_, hint: str = "r") -> Var:
        """Fresh Var + ``AllocTensor`` Call + ``_Bind``.

        Fresh Var + ``AllocTensor`` Call + ``_Bind`` — the alloc-and-bind
        boilerplate shared by every buffer-introducing lowering handler.
        """
        r = self._fresh(type_, hint=hint)
        alloc_r = Call(type=r.type, target=AllocTensorOp(tensor_type=r.type), args=())
        self._items.append(_Bind(var=r, value=alloc_r))
        return r

    def _fork(self, *, bind: "dict[int, Var]") -> "_Lowerer":
        """Fork lowering state for a nested grid-region body.

        Share dispatch context and shape-parameter collection by reference while
        copying accumulated caches, tuple parts, carry roots, and name counter.
        *bind* seeds additional expression-to-variable entries.
        """
        sub = _Lowerer(
            dispatch_groups=self._dispatch_groups,
            mangled_registry=self._mangled_registry,
            caller_fn=self._caller_fn,
            shape_param_names=self._shape_param_names,
        )
        sub._name_counter = self._name_counter
        sub._cache.update(self._cache)
        sub._cache.update(bind)
        sub._tuple_parts.update(self._tuple_parts)
        sub._carry_init.update(self._carry_init)
        return sub

    def _close(self, sub: "_Lowerer") -> Sequential:
        """Close.

        Sync the name counter forward from a forked sub-lowerer and fold
        its collected items into a nested ``LetStmt``/``Sequential`` chain.
        """
        self._name_counter = sub._name_counter
        return _fold_items_to_sequential(sub._items)






    def lower(self, expr: Expr) -> Var:
        """Recursively lower a nested HIR sub-expression to its TIR Var."""
        return self.lower_expr(expr)

    def fresh(self, type_, hint: str = "v") -> Var:
        """Allocate a fresh Var name (no binding emitted)."""
        return self._fresh(type_, hint=hint)

    def alloc(self, type_, hint: str = "r") -> Var:
        """Fresh Var + ``AllocTensor`` Call + binding (see ``_alloc``)."""
        return self._alloc(type_, hint=hint)

    def emit(self, stmt: Stmt) -> None:
        """Append a raw Stmt to the pending item sequence."""
        self._items.append(stmt)

    def emit_bind(self, var: Var, value: Expr) -> None:
        """Append a let-binding ``var = value`` to the pending item sequence."""
        self._items.append(_Bind(var=var, value=value))

    def lower_expr(self, expr: Expr) -> Var:
        key = id(expr)
        if key in self._cache:
            return self._cache[key]
        if isinstance(expr, Var):
            self._cache[key] = expr
            return expr
        if isinstance(expr, Constant):



            const_type = expr.type
            if const_type.storage is StorageKind.UMAT:
                const_type = TensorType(
                    shape=const_type.shape,
                    dtype=const_type.dtype,
                    layout=const_type.layout,
                    storage=StorageKind.RMEM,
                )
            r = self._alloc(const_type, hint="c")
            self._items.append(_eval_call(Fill(), (r, expr)))
            self._cache[key] = r
            return r
        if isinstance(expr, GridRegionExpr):
            _reject_partial_window(expr)



            if expr.carried_args:
                return self._lower_grid_region_carry(expr)

            iv = expr.induction_var
            grid_ty = expr.type


            M = grid_ty.shape[0] if grid_ty.shape else 1


            out_type = TensorType(
                shape=grid_ty.shape,
                dtype=grid_ty.dtype,
                layout=grid_ty.layout,
                storage=grid_ty.storage,
            )
            out_var = self._alloc(out_type, hint="grid_out")


            sub = self._fork(bind={id(iv): iv})


            body_result = sub.lower_expr(expr.body)


            view_shape = body_result.type.shape
            out_view_layout = TensorView.layout_for_slice(
                src_shape=tuple(out_var.type.shape),
                axis=0,
                sliced_shape=view_shape,
            )
            out_view_op = TensorView(layout=out_view_layout)
            out_view_type = TensorType(
                shape=view_shape,
                dtype=out_var.type.dtype,
                layout=out_view_layout,
                storage=out_var.type.storage,
            )
            element_start = simplify_dim(
                DimMul,
                (iv, shape_numel_upper_bound(tuple(view_shape))),
            )
            out_view_call = Call(
                type=out_view_type,
                target=out_view_op,
                args=(out_var, element_start),
            )
            out_view_var = sub._fresh(out_view_type, hint="ov")
            sub._items.append(_Bind(var=out_view_var, value=out_view_call))
            sub._items.append(_eval_call(Copy(), (body_result, out_view_var)))

            body_seq = self._close(sub)


            i32_scalar = TensorType.scalar(dtype=DType.i32, storage=StorageKind.RMEM)
            start_val = Constant(value=0, type=i32_scalar)
            stop_val = Constant(value=M, type=i32_scalar)
            step_val = Constant(value=1, type=i32_scalar)
            for_loop = For(
                induction_var=iv, start=start_val, stop=stop_val,
                step=step_val, body=body_seq,
            )
            self._items.append(for_loop)
            self._cache[key] = out_var
            return out_var
        if not isinstance(expr, Call):
            raise TypeError(
                f"demo lowering: unexpected Expr {type(expr).__name__}"
            )
        target = expr.target
        handler = hir_lowering_registry.lookup(type(target))
        if handler is None:
            raise TypeError(
                f"hir_to_tir: no lowering registered for Op "
                f"{type(target).__name__}"
            )



        result = handler(self, target, expr)
        self._cache[key] = result
        return result

    def _lower_grid_region_carry(self, expr: GridRegionExpr) -> Var:
        """Lower a loop-carried ``GridRegionExpr`` (accumulator loop).

        Each phi is materialised as a mutable buffer initialised from its
        ``init_args`` value before the loop; each iteration lowers ``body`` with
        the phi Var bound to that buffer and copies the ``yield_values`` result
        back into it. The buffer is the loop's value. A yield that already IS the
        accumulator (an in-place op returning its dst — e.g. ``insert_slice`` /
        ``cache_update``) needs no copy-back, so the single carried buffer is
        reused across iterations with no replacement allocation in the body.
        """
        key = id(expr)
        start, stop, step = expr.start, expr.extent, expr.step
        if not all(isinstance(b, int) for b in (start, stop, step)):
            raise NotImplementedError(
                f"hir_to_tir: GridRegionExpr carry lowering needs static int "
                f"loop bounds, got start={start!r}, extent={stop!r}, "
                f"step={step!r}"
            )
        iv = expr.induction_var




        acc_vars: list[Var] = []
        for init_arg in expr.init_args:
            init_var = self.lower_expr(init_arg)
            if init_var.type.storage == StorageKind.GMEM:




                acc_vars.append(init_var)
                continue
            acc_var = self._alloc(init_var.type, hint="acc")
            self._items.append(_eval_call(Copy(), (init_var, acc_var)))
            acc_vars.append(acc_var)


        sub = self._fork(bind={id(iv): iv})



        for phi, init_arg in zip(expr.carried_args, expr.init_args):
            self._carry_init[id(phi)] = init_arg
            sub._carry_init[id(phi)] = init_arg
        for phi, acc_var in zip(expr.carried_args, acc_vars):
            sub._cache[id(phi)] = acc_var
        sub.lower_expr(expr.body)


        yield_vars = [sub.lower_expr(y) for y in expr.yield_values]
        for yield_var, acc_var in zip(yield_vars, acc_vars):
            if yield_var is not acc_var:
                sub._items.append(_eval_call(Copy(), (yield_var, acc_var)))

        body_seq = self._close(sub)

        i32_scalar = TensorType.scalar(dtype=DType.i32, storage=StorageKind.RMEM)
        for_loop = For(
            induction_var=iv,
            start=Constant(value=start, type=i32_scalar),
            stop=Constant(value=stop, type=i32_scalar),
            step=Constant(value=step, type=i32_scalar),
            body=body_seq,
        )
        self._items.append(for_loop)


        self._tuple_parts[key] = acc_vars
        self._cache[key] = acc_vars[0]
        return acc_vars[0]

    def _reshard_cross_cta_sync(self, expr) -> "Var | None":
        """Fence a param-rooted global buffer before cross-CTA reownership.

        Lower the producing chain, fence the grid, and return its root only when
        source and destination use different CTA shard ownership. Return ``None``
        otherwise. This owns synchronization, not data redistribution, and is
        shared by intermediate and output-sink reshard paths.
        """
        dst_sl = expr.type.layout
        src_expr = expr.args[0]
        if (
            isinstance(dst_sl, ShardLayout)
            and expr.type.storage == StorageKind.GMEM
            and not isinstance(src_expr, Var)
            and getattr(getattr(src_expr, "type", None), "storage", None)
            == StorageKind.GMEM
            and isinstance(getattr(src_expr.type, "layout", None), ShardLayout)


            and src_expr.type.layout != dst_sl
            and all(t.name == "cta" for t in dst_sl.mesh.topologies)
        ):
            root = self._param_alias_root(src_expr)
            if root is not None:


                self.lower_expr(src_expr)
                self._items.append(_eval_call(TirSync(mesh=dst_sl.mesh), ()))
                return root
        return None

    def _param_alias_root(self, expr, _depth: int = 0):
        """Walk Reshard / loop-carry chains to the underlying kernel-param Var, or ``None``.

        Walk Reshard / loop-carry chains to the underlying kernel-param Var
        (the gmem alias root), or ``None``. Used by the cross-CTA
        sync-then-reshard rule in ``_lower_reshard``.
        """
        class _AliasVisitor(ExprVisitor[Var | None]):
            def __init__(self, owner) -> None:
                super().__init__()
                self.owner = owner
                self.depth = _depth

            def _recurse(self, value):
                self.depth += 1
                try:
                    return self.visit(value)
                finally:
                    self.depth -= 1

            def dispatch_visit(self, value):
                if self.depth > 64:
                    return None
                return ExprFunctor.dispatch_visit(self, value)

            def visit_Var(self, value: Var) -> Var | None:
                initial = self.owner._carry_init.get(id(value))
                if initial is not None:
                    return self._recurse(initial)
                return value if self.owner._cache.get(id(value)) is value else None

            def visit_GridRegionExpr(self, value: GridRegionExpr) -> Var | None:
                for initial in value.init_args:
                    result = self._recurse(initial)
                    if result is not None:
                        return result
                return None

            def visit_Call(self, value: Call) -> Var | None:
                if isinstance(value.target, Reshard):
                    return self._recurse(value.args[0])
                return None

            def default_visit(self, value) -> Var | None:
                return None

        return _AliasVisitor(self).visit(expr) if expr is not None else None


    def _lower_hir_call(self, call: Call, callee_hir: HirFunction) -> Var:
        """Lower ``Call(target=HirFunction)`` into a ``tir.DispatchCall``.

        Only invoked when the callee's overload group has at least one
        non-empty specialization; the static-callee path is intentionally
        not supported here (no static-callee HIR Call exists in the v0
        regression suite, and adding one would over-extend the regression
        suite).
        """
        group = self._dispatch_groups.get(callee_hir.name, ())
        if not group:
            raise TypeError(
                f"HIR Call to {callee_hir.name!r}: callee has no "
                f"specializations and is not in a dispatch group"
            )

        arg_vars: list[Var] = [self.lower_expr(a) for a in call.args]

        out_type = call.type
        out_var = self._alloc(out_type, hint="cr")





        first_variant = group[0]
        first_pat = first_variant.specializations[0]
        if not isinstance(first_pat, DimVarRangePat):
            raise TypeError(
                f"HIR Call to {callee_hir.name!r}: only DimVarRangePat "
                f"dispatch is supported in v0"
            )
        dim_name = first_pat.dim_var
        loc = locate_dim_var(first_variant.params, dim_name)
        if loc is None:
            raise TypeError(
                f"HIR Call to {callee_hir.name!r}: dispatch DimVar "
                f"{dim_name!r} not found in callee signature"
            )
        param_index, axis = loc


        if param_index >= len(call.args):
            raise TypeError(
                f"HIR Call to {callee_hir.name!r}: arg index {param_index} "
                f"out of range for {len(call.args)} args"
            )
        arg_ty = call.args[param_index].type
        if not isinstance(arg_ty, TensorType) or axis >= len(arg_ty.shape):
            raise TypeError(
                f"HIR Call to {callee_hir.name!r}: arg[{param_index}] "
                f"shape does not expose axis {axis}"
            )
        dim_entry = arg_ty.shape[axis]
        caller_range = self._resolve_caller_range(dim_entry, callee_hir.name)
        c_lo, c_hi = caller_range

        reachable: list[tuple[HirFunction, DimVarRangePat]] = []
        for variant in group:
            pat = variant.specializations[0]
            if not isinstance(pat, DimVarRangePat):
                continue

            if pat.lo < c_hi and c_lo < pat.hi:
                reachable.append((variant, pat))
        if not reachable:
            raise TypeError(
                f"HIR Call to {callee_hir.name!r}: empty reachable "
                f"specialization set; caller range [{c_lo}, {c_hi}) "
                f"does not intersect any callee specialization range"
            )




        subject_param = arg_vars[param_index]





        if (
            self._caller_fn is None
            or not any(subject_param is p for p in self._caller_fn.params)
        ):
            raise TypeError(
                f"HIR Call to {callee_hir.name!r}: dispatch subject must "
                f"be a caller param (caller-arg expression lowering is "
                f"not supported in v0 sub-call dispatch)"
            )
        self._shape_param_names.add((subject_param.name, axis))
        scalar_i32 = TensorType.scalar(dtype=DType.i32, storage=StorageKind.RMEM)
        subject = ShapeOf(type=scalar_i32, param=subject_param, axis=axis)
        case_patterns = tuple((pat,) for _, pat in reachable)
        case_calls: list[Evaluate] = []
        for variant, pat in reachable:
            mangled_name = _mangle_variant_name(variant)
            mangled_pf = self._mangled_registry.get(mangled_name)
            if mangled_pf is None:
                raise RuntimeError(
                    f"HIR Call to {callee_hir.name!r}: mangled callee "
                    f"{mangled_name!r} not pre-lowered"
                )










            call_args: list = [*arg_vars, out_var]
            head_count = len(variant.params) + mangled_pf.output_count
            trailing = mangled_pf.params[head_count:]
            for tp in trailing:
                parsed = parse_shape_var_name(tp.name)
                if parsed is None:
                    raise RuntimeError(
                        f"HIR Call to {callee_hir.name!r}: mangled callee "
                        f"{mangled_pf.name!r} has trailing param {tp.name!r} "
                        f"that is not a <base>_shape_<axis> shape scalar"
                    )
                base_name, axis_int = parsed
                user_idx = next(
                    (i for i, p in enumerate(variant.params) if p.name == base_name),
                    None,
                )
                if user_idx is None:
                    raise RuntimeError(
                        f"HIR Call to {callee_hir.name!r}: mangled callee "
                        f"{mangled_pf.name!r} trailing scalar references user "
                        f"param {base_name!r} that is not in the callee "
                        f"user-param list"
                    )
                caller_arg_var = arg_vars[user_idx]
                if (
                    self._caller_fn is None
                    or not any(caller_arg_var is p for p in self._caller_fn.params)
                ):
                    raise TypeError(
                        f"HIR Call to {callee_hir.name!r}: forwarded "
                        f"shape-scalar arg for {tp.name!r} requires the "
                        f"caller-side Var to be a caller param "
                        f"(got {caller_arg_var.name!r})"
                    )
                self._shape_param_names.add((caller_arg_var.name, axis_int))
                call_args.append(
                    ShapeOf(
                        type=scalar_i32, param=caller_arg_var, axis=axis_int,
                    )
                )
            case_calls.append(symbol_call(mangled_pf, call_args))
        dispatch = DispatchCall(
            callee_name=callee_hir.name,
            subjects=(subject,),
            case_patterns=case_patterns,
            case_calls=tuple(case_calls),
            fallback=Sequential(body=(Abort(),)),
        )
        self._items.append(dispatch)
        self._cache[id(call)] = out_var
        return out_var

    def _resolve_caller_range(
        self, dim_entry: object, callee_name: str
    ) -> tuple[int, int]:
        """Resolve a caller-side half-open ``[lo, hi)`` range for a shape entry.

        Static int ``k`` → ``[k, k+1)`` (the single value). A
        ``DimVar(name, lo, hi)`` carries its half-open bounds directly.
        Anything else is a compile-time error.
        """
        from tilefoundry.ir.types.dim import DimVar  # noqa: PLC0415 — avoid cycle

        if isinstance(dim_entry, int) and not isinstance(dim_entry, bool):
            return (dim_entry, dim_entry + 1)
        if isinstance(dim_entry, DimVar):
            return (dim_entry.lo, dim_entry.hi)
        raise TypeError(
            f"HIR Call to {callee_name!r}: cannot resolve caller-side "
            f"range from dim entry {dim_entry!r} — only static int dims "
            f"or bounded DimVar(name, lo, hi) entries are supported"
        )








@register_hir_lowering(Reshard)
def _lower_reshard(ctx: "_Lowerer", target, expr) -> Var:
    """Lower a reshard using the post-inference destination layout.

    Reuse a materialized source layout or derive shared-engine strides for a
    plain boundary input. See [passes §7.1](docs/spec/passes.md#71-hirtotirpass)
    and [hir §1.1](docs/spec/hir.md#11-function).
    """
    src_override = ctx._reshard_cross_cta_sync(expr)
    src = src_override if src_override is not None else ctx.lower(expr.args[0])
    src_ty = src.type











    sl = expr.type.layout
    dst_shape = expr.type.shape
    view_src = src
    if isinstance(src_ty.layout, ShardLayout):
        src_sl = src_ty.layout





        ptr_call = Call(type=src_ty, target=PtrOf(), args=(src,))
        view_src = ctx.fresh(src_ty, hint="ptr")
        ctx.emit_bind(view_src, ptr_call)
    elif isinstance(sl, ShardLayout):






        if _is_full_layout(sl.layout):
            src_strides = tuple(int(s) for s in sl.layout.strides)
        else:
            src_strides = _shared_engine_strides(sl)
        src_sl = ShardLayout(
            layout=_Layout(
                shape=sl.layout.shape,
                strides=src_strides,
            ),
            attrs=sl.attrs,
            mesh=sl.mesh,
        )
    else:
        src_sl = sl


    tv = TensorView(layout=src_sl)
    tv_type = TensorType(
        shape=dst_shape,
        dtype=src_ty.dtype,
        layout=src_sl,
        storage=src_ty.storage,
    )
    tv_call = Call(type=tv_type, target=tv, args=(view_src,))
    sv = ctx.fresh(tv_type, hint="sv")
    ctx.emit_bind(sv, tv_call)

    if target.storage is None:

        return sv



    per_shard_shape = list(sl.layout.shape)
    mesh_shape = sl.mesh.layout.shape
    attr_idx = 0
    for a in sl.attrs:
        if isinstance(a, Split) and attr_idx < len(mesh_shape):
            mext = mesh_shape[attr_idx]
            if mext is None:




                per_shard_shape[a.axis] = 1
            else:





                per_shard_shape[a.axis] = max(
                    1, per_shard_shape[a.axis] // mext
                )
        attr_idx += 1
    per_shard_shape = tuple(per_shard_shape)




    dst_type = TensorType(
        shape=per_shard_shape,
        dtype=src_ty.dtype,
        layout=sl,
        storage=target.storage,
    )
    dst = ctx.alloc(dst_type, hint="t")
    ctx.emit(_eval_call(Copy(), (sv, dst)))
    return dst






@register_hir_lowering(HirMmaSM80_16x8x16)
@register_hir_lowering(HirWgmma_SM90)
def _lower_mma(ctx: "_Lowerer", target, expr) -> Var:
    name = type(target).__name__
    raise ValueError(
        f"HirToTirPass: {name} HIR compile route is unsupported; use the "
        "independent handwritten TIR MMA atom/runtime surface"
    )


@register_hir_lowering(HirReLU)
def _lower_relu(ctx: "_Lowerer", target, expr) -> Var:
    x = ctx.lower(expr.args[0])


    r = ctx.alloc(x.type, hint="r")
    ctx.emit(_eval_call(TirReLU(), (x, r)))
    return r



@register_hir_lowering(HirBinary)
def _lower_binary(ctx: "_Lowerer", target, expr) -> Var:
    lhs = ctx.lower(expr.args[0])
    rhs = ctx.lower(expr.args[1])





    out_shape = lhs.type.shape if len(lhs.type.shape) >= len(rhs.type.shape) else rhs.type.shape




    out_storage = (
        expr.type.storage
        if expr.type.storage is not StorageKind.UMAT
        else StorageKind.RMEM
    )
    out_type = TensorType(
        shape=out_shape,
        dtype=expr.type.dtype,
        layout=lhs.type.layout if lhs.type.layout is not None else rhs.type.layout,
        storage=out_storage,
    )
    r = ctx.alloc(out_type, hint="r")
    ctx.emit(_eval_call(TirBinary(kind=target.kind), (lhs, rhs, r)))
    return r


@register_hir_lowering(HirUnary)
def _lower_unary(ctx: "_Lowerer", target, expr) -> Var:
    x = ctx.lower(expr.args[0])


    out_type = TensorType(
        shape=x.type.shape,
        dtype=expr.type.dtype,
        layout=x.type.layout,
        storage=x.type.storage,
    )
    r = ctx.alloc(out_type, hint="r")
    ctx.emit(_eval_call(TirUnary(kind=target.kind), (x, r)))
    return r


@register_hir_lowering(HirReshape)
def _lower_reshape(ctx: "_Lowerer", target, expr) -> Var:



    x = ctx.lower(expr.args[0])

    ptr_op = PtrOf()
    ptr_ty = x.type
    ptr_call = Call(type=ptr_ty, target=ptr_op, args=(x,))
    ptr_var = ctx.fresh(ptr_ty, hint="ptr")
    ctx.emit_bind(ptr_var, ptr_call)

    out_ty = TensorType(
        shape=target.new_shape,
        dtype=x.type.dtype,
        layout=x.type.layout,
        storage=x.type.storage,
    )
    tv = TensorView(layout=out_ty.layout, shape=target.new_shape)
    tv_call = Call(type=out_ty, target=tv, args=(ptr_var,))
    r = ctx.fresh(out_ty, hint="r")
    ctx.emit_bind(r, tv_call)
    return r


@register_hir_lowering(HirClamp)
def _lower_clamp(ctx: "_Lowerer", target, expr) -> Var:
    x = ctx.lower(expr.args[0])
    r = ctx.alloc(x.type, hint="r")
    ctx.emit(_eval_call(
        TirClamp(min_val=target.min_val, max_val=target.max_val), (x, r)))
    return r


@register_hir_lowering(HirCast)
def _lower_cast(ctx: "_Lowerer", target, expr) -> Var:
    x = ctx.lower(expr.args[0])
    out_type = TensorType(
        shape=x.type.shape,
        dtype=target.dtype,
        layout=x.type.layout,
        storage=x.type.storage,
    )
    r = ctx.alloc(out_type, hint="r")
    ctx.emit(_eval_call(TirUnary(kind=UnaryKind.CAST), (x, r)))
    return r


@register_hir_lowering(HirTupleGetItem)
def _lower_tuple_get_item(ctx: "_Lowerer", target, expr) -> Var:



    src = expr.args[0]
    ctx.lower(src)
    parts = ctx._tuple_parts.get(id(src))
    if parts is None:
        raise TypeError(
            "hir_to_tir: TupleGetItem on a producer with no lowered tuple "
            f"fields ({type(src).__name__})"
        )
    return parts[target.index]


@register_hir_lowering(HirFullLike)
def _lower_full_like(ctx: "_Lowerer", target, expr) -> Var:





    ty = expr.type
    storage = (
        ty.storage if ty.storage is not StorageKind.UMAT else StorageKind.RMEM
    )
    try:
        tmpl = ctx.lower(expr.args[0])
    except Exception:
        tmpl = None
    if tmpl is not None:
        out_type = TensorType(
            shape=tuple(tmpl.type.shape),
            dtype=ty.dtype,
            layout=tmpl.type.layout,
            storage=(
                tmpl.type.storage
                if tmpl.type.storage is not StorageKind.UMAT
                else storage
            ),
        )
    else:


        local_shape = (
            shard_layout_local_shape(ty.layout)
            if isinstance(ty.layout, ShardLayout)
            else tuple(ty.shape)
        )
        out_type = TensorType(
            shape=tuple(local_shape),
            dtype=ty.dtype,
            layout=ty.layout,
            storage=storage,
        )
    r = ctx.alloc(out_type, hint="c")
    fill_value = Constant(
        value=target.value,
        type=TensorType(
            shape=(), dtype=ty.dtype, layout=None, storage=StorageKind.RMEM
        ),
    )
    ctx.emit(_eval_call(Fill(), (r, fill_value)))
    return r


@register_hir_lowering(HirCacheUpdate)
def _lower_cache_update(ctx: "_Lowerer", target, expr) -> Var:
    cache = ctx.lower(expr.args[0])
    cur = ctx.lower(expr.args[1])


    new = ctx.lower(expr.args[3])
    cache_shape = tuple(cache.type.shape)
    new_shape = tuple(expr.args[3].type.shape)
    if new_shape[1] != 1 or cache_shape[0] != 1:
        raise NotImplementedError(
            "cache_update lowering: v0 supports B == 1 and S_CAP == 1 "
            f"(single-row decode write); got cache {cache_shape}, "
            f"new {new_shape}"
        )


    view_shape = (new_shape[0], 1, *cache_shape[2:])
    cstr = c_order_strides(tuple(int(d) for d in cache_shape))
    view_layout = _Layout(shape=view_shape, strides=cstr)
    tv_type = TensorType(
        shape=view_shape,
        dtype=cache.type.dtype,
        layout=view_layout,
        storage=cache.type.storage,
    )
    i32 = TensorType.scalar(dtype=DType.i32, storage=StorageKind.RMEM)
    zero = Constant(value=0, type=i32)
    tv_call = Call(
        type=tv_type,
        target=TensorView(layout=view_layout),
        args=(cache, zero, cur, zero, zero),
    )
    sv = ctx.fresh(tv_type, hint="sv")
    ctx.emit_bind(sv, tv_call)


    ctx.emit(_eval_call(Copy(), (new, sv)))
    return cache


def _insert_slice_coord(ctx: "_Lowerer", off_expr):
    """The scalar window index for an in-place ``insert_slice``.

    The absolute element coordinate where the update window starts. A
    compile-time offset folds to a ``Constant`` literal; dim arithmetic over the
    induction variable is an address the emitter computes, so it is carried
    through; a runtime scalar offset lowers to its scalar Var, whose single
    element is read at the coordinate site. A coordinate an *op* computes is
    refused: materializing it gives a buffer, and a buffer is not a scalar index.
    """
    i32 = TensorType.scalar(dtype=DType.i32, storage=StorageKind.RMEM)
    if isinstance(off_expr, Constant):
        val = off_expr.value
        elem = int(val[0]) if isinstance(val, (list, tuple)) else int(val)
        return Constant(value=elem, type=i32)
    if _is_coordinate_arithmetic(off_expr):
        return off_expr
    if isinstance(off_expr, Call):
        raise NotImplementedError(
            f"hir_to_tir: a window coordinate computed by "
            f"{type(off_expr.target).__name__} has no lowering -- it would be "
            f"materialized into a buffer and that buffer used where a scalar "
            f"index belongs. A coordinate is a literal, a scalar operand, or a "
            f"compile-time offset off one (`i + C`)"
        )
    return ctx.lower(off_expr)


class _CoordinateArithmeticVisitor(ExprVisitor[bool]):
    def visit_Constant(self, expr: Constant) -> bool:
        return True

    def visit_Var(self, expr: Var) -> bool:
        return True

    def visit_Call(self, expr: Call) -> bool:
        return is_dim_op_call(expr) and all(self.visit(arg) for arg in expr.args)

    def default_visit(self, expr) -> bool:
        return False


def _is_coordinate_arithmetic(expr) -> bool:
    """Whether *expr* is dim arithmetic over literals and scalar Vars.

    Such an offset is an address the emitter computes, not a value a tensor op
    produces, so lowering carries it through to the coordinate site.
    """
    if not is_dim_op_call(expr):
        return False
    if isinstance(expr, Call):
        return _CoordinateArithmeticVisitor().visit(expr)
    return False


@register_hir_lowering(HirSlice)
def _lower_slice(ctx: "_Lowerer", target, expr) -> Var:
    """Lower a unit-stride HIR slice to an absolute-coordinate tensor view."""
    source = ctx.lower(expr.args[0])
    starts = expr.args[1]
    if not isinstance(starts, Tuple):
        raise TypeError("hir_to_tir: Slice starts must be a Tuple")
    if any(s != 1 for s in target.strides):
        raise NotImplementedError("hir_to_tir: Slice lowering supports unit strides only")
    coords = tuple(_insert_slice_coord(ctx, start) for start in starts.elements)
    view_shape = tuple(expr.type.shape)
    view_layout = TensorView.layout_for_slice_nd(
        src_shape=tuple(source.type.shape), sliced_shape=view_shape
    )
    view_type = TensorType(
        shape=view_shape,
        dtype=source.type.dtype,
        layout=view_layout,
        storage=source.type.storage,
    )
    view_call = Call(
        type=view_type,
        target=TensorView(layout=view_layout),
        args=(source, *coords),
    )
    result = ctx.fresh(view_type, hint="slice")
    ctx.emit_bind(result, view_call)
    return result


@register_hir_lowering(HirInsertSlice)
def _lower_insert_slice(ctx: "_Lowerer", target, expr) -> Var:
    dst = ctx.lower(expr.args[0])
    upd = ctx.lower(expr.args[1])






    off_expr = expr.args[2]
    upd_shape = tuple(upd.type.shape)
    if isinstance(off_expr, Tuple):
        coords = tuple(_insert_slice_coord(ctx, el) for el in off_expr.elements)
    else:
        coords = (_insert_slice_coord(ctx, off_expr),)



    if isinstance(upd.type.layout, ShardLayout):
        win_layout = upd.type.layout
    elif isinstance(off_expr, Tuple):
        win_layout = TensorView.layout_for_slice_nd(
            src_shape=tuple(dst.type.shape), sliced_shape=upd_shape
        )
    else:
        win_layout = TensorView.layout_for_slice(
            src_shape=tuple(dst.type.shape), axis=0, sliced_shape=upd_shape
        )
    win_type = TensorType(
        shape=upd_shape,
        dtype=dst.type.dtype,
        layout=win_layout,
        storage=dst.type.storage,
    )
    win_call = Call(
        type=win_type, target=TensorView(layout=win_layout), args=(dst, *coords)
    )
    win = ctx.fresh(win_type, hint="isv")
    ctx.emit_bind(win, win_call)
    ctx.emit(_eval_call(Copy(), (upd, win)))
    return dst



@register_hir_lowering(HirIndexSelect)
def _lower_index_select(ctx: "_Lowerer", target, expr) -> Var:
    if tuple(expr.args[1].type.shape) != (1,):
        raise NotImplementedError(
            "IndexSelect HIR-to-TIR lowering supports only a one-element index"
        )
    rank = len(expr.args[0].type.shape)
    dim = target.dim + rank if target.dim < 0 else target.dim
    if any(extent != 1 for extent in expr.args[0].type.shape[:dim]):
        raise NotImplementedError(
            "IndexSelect HIR-to-TIR view lowering requires unit leading dims"
        )
    x = ctx.lower(expr.args[0])
    index = ctx.lower(expr.args[1])

    view_shape = expr.type.shape
    view_layout = _Layout(
        shape=tuple(view_shape),
        strides=tuple(c_order_strides(tuple(x.type.shape))),
    )
    source_stride = c_order_strides(tuple(x.type.shape))[dim]
    element_start = simplify_dim(DimMul, (index, source_stride))
    tv = TensorView(layout=view_layout)
    tv_type = TensorType(
        shape=view_shape,
        dtype=expr.type.dtype,
        layout=view_layout,
        storage=x.type.storage,
    )
    tv_call = Call(type=tv_type, target=tv, args=(x, element_start))
    sv = ctx.fresh(tv_type, hint="sv")
    ctx.emit_bind(sv, tv_call)
    return sv


@register_hir_lowering(HirReduce)
def _lower_reduce(ctx: "_Lowerer", target, expr) -> Var:
    x = ctx.lower(expr.args[0])
    axes = target.axes







    keepdim = target.keepdim
    new_shape = list(x.type.shape)
    for a in sorted(axes, reverse=True):
        if keepdim:
            new_shape[a] = 1
        else:
            new_shape.pop(a)
    out_type = TensorType(
        shape=tuple(new_shape),
        dtype=x.type.dtype,
        layout=getattr(expr.type, "layout", x.type.layout),
        storage=x.type.storage,
    )
    r = ctx.alloc(out_type, hint="r")








    tir_args: tuple = (x, r)







    ws_size, ws_dtype, lane_reduced = _analyze_cross_warp_workspace(
        expr.args[0].type, axes
    )
    if ws_size > 0:



        n_cells = 1
        for dim in r.type.shape:
            if isinstance(dim, int):
                n_cells *= dim
        ws_size *= max(1, n_cells)
        if not lane_reduced:

            ws_size *= 32
        ws_type = TensorType(
            shape=(ws_size,),
            dtype=ws_dtype,
            layout=None,
            storage=StorageKind.SMEM,
        )
        ws = ctx.alloc(ws_type, hint="ws")
        tir_args = (x, r, ws)

    reduce_op = TirReduce(axes=axes, kind=target.kind)
    ctx.emit(_eval_call(reduce_op, tir_args))
    return r


@register_hir_lowering(HirFunction)
def _lower_hir_function(ctx: "_Lowerer", target, expr) -> Var:
    return ctx._lower_hir_call(expr, target)


def _mangle_variant_name(variant: HirFunction) -> str:
    """Mangle a dispatch variant's symbol from its single Pattern."""
    if len(variant.specializations) != 1:
        raise TypeError(
            f"variant {variant.name!r}: expected exactly one "
            f"specialization, got {len(variant.specializations)}"
        )
    pat = variant.specializations[0]
    if not isinstance(pat, DimVarRangePat):
        raise TypeError(
            f"variant {variant.name!r}: only DimVarRangePat is supported "
            f"for v0 specialization mangling"
        )
    return f"{variant.name}${pat.dim_var}${pat.lo}_{pat.hi}"


def _fold_items_to_sequential(items: list[_Item]) -> Sequential:
    """Turn a flat item list into a nested ``LetStmt`` chain wrapped in a ``Sequential``.

    Turn a flat item list into a nested ``LetStmt`` chain wrapped in a
    ``Sequential``.
    """
    def fold(i: int) -> list[Stmt]:
        out: list[Stmt] = []
        while i < len(items):
            item = items[i]
            if isinstance(item, _Bind):
                inner = fold(i + 1)
                out.append(
                    LetStmt(
                        var=item.var,
                        value=item.value,
                        body=Sequential(body=tuple(inner)),
                    )
                )
                return out
            out.append(item)
            i += 1
        return out

    return Sequential(body=tuple(fold(0)))


def _lower_single_output(
    lo: "_Lowerer",
    body_expr: Expr,
    out_var: Var,
) -> None:
    """Lower a single HIR body expression and copy the result to *out_var*."""
    if (
        isinstance(body_expr, Call)
        and isinstance(body_expr.target, Reshard)
    ):




        override = lo._reshard_cross_cta_sync(body_expr)
        inner = override if override is not None else lo.lower_expr(body_expr.args[0])
        inner_ty = inner.type




        sl = getattr(body_expr.type, "layout", None)
        if sl is None:
            sl = body_expr.target.layout
        if sl is None:
            sl = getattr(inner_ty, "layout", None)



        tv = TensorView(layout=sl)
        tv_type = TensorType(
            shape=out_var.type.shape,
            dtype=out_var.type.dtype,
            layout=sl,
            storage=out_var.type.storage,
        )
        tv_call = Call(type=tv_type, target=tv, args=(out_var,))
        sv = lo._fresh(tv_type, hint="sv")
        lo._items.append(_Bind(var=sv, value=tv_call))
        lo._items.append(_eval_call(Copy(), (inner, sv)))
    else:
        src = lo.lower_expr(body_expr)
        lo._items.append(_eval_call(Copy(), (src, out_var)))


def _lower_function(
    fn: HirFunction,
    *,
    target: Target,
    cta_mesh: Mesh | None,
    thread_mesh: Mesh | None,
    cta_var_name: str = "block",
    thread_var_name: str = "thread",
    out_var_name: str = "out",
    override_name: str | None = None,
    dispatch_groups: "dict[str, tuple[HirFunction, ...]] | None" = None,
    mangled_registry: "dict[str, PrimFunction] | None" = None,
) -> PrimFunction:
    """Materialise the HIR `Function -> tensor` as an explicit-output-param ``PrimFunction``.

    Materialise the HIR `Function(params) -> tensor` as an
    explicit-output-param ``PrimFunction``. The function-end
    sink — the outermost Reshard write to global — is rewritten as a
    ``Copy(<reg/shared result>, out)`` into the new ``out`` parameter
    instead of allocating a fresh global tensor.
    """
    is_tuple_return = isinstance(fn.return_type, TupleType)
    if is_tuple_return:
        _field_types = fn.return_type.fields
    else:
        _field_types = (fn.return_type,)

    out_vars: list[Var] = []
    for i, _ft in enumerate(_field_types):
        _flat_ret_ty = TensorType(
            shape=_ft.shape,
            dtype=_ft.dtype,
            layout=None,
            storage=_ft.storage,
        )
        _name = out_var_name if len(_field_types) == 1 else f"{out_var_name}{i}"
        out_vars.append(Var(type=_flat_ret_ty, name=_name))

    shape_param_names: set[tuple[str, int]] = set()






    from tilefoundry.ir.types.dim import DimVar  # noqa: PLC0415 — avoid cycle
    for p in fn.params:
        ty = getattr(p, "type", None)
        if not isinstance(ty, TensorType):
            continue
        for axis, dim in enumerate(ty.shape):
            if isinstance(dim, DimVar):
                shape_param_names.add((p.name, axis))
    lo = _Lowerer(
        dispatch_groups=dispatch_groups,
        mangled_registry=mangled_registry,
        caller_fn=fn,
        shape_param_names=shape_param_names,
    )
    for p in fn.params:
        lo._cache[id(p)] = p

    body_expr = fn.body
    if is_tuple_return:
        if not isinstance(body_expr, Tuple):
            raise TypeError(
                "TupleType return requires Tuple body expr, "
                f"got {type(body_expr).__name__}"
            )
        _elements = body_expr.elements
    else:
        _elements = (body_expr,)

    for _elem, _out_var in zip(_elements, out_vars):
        _lower_single_output(lo, _elem, _out_var)

    inner_seq = _fold_items_to_sequential(lo._items)




    scoped: Stmt = inner_seq
    if thread_mesh is not None:
        thread_binding = Var(type=fn.body.type, name=thread_var_name)
        scoped = MeshScope(
            mesh=thread_mesh, binding=thread_binding, body=scoped
        )
        scoped = Sequential(body=(scoped,))
    if cta_mesh is not None:
        cta_binding = Var(type=fn.body.type, name=cta_var_name)
        scoped = MeshScope(
            mesh=cta_mesh, binding=cta_binding, body=scoped
        )
        body = Sequential(body=(scoped, Return()))
    elif thread_mesh is not None:


        body = Sequential(body=(scoped, Return()))
    else:

        body = Sequential(body=(*inner_seq.body, Return()))




    shape_params: list[Var] = []
    scalar_i32_ty = TensorType.scalar(dtype=DType.i32, storage=StorageKind.RMEM)
    for pname, axis in sorted(shape_param_names):
        shape_params.append(
            Var(type=scalar_i32_ty, name=shape_var_name(pname, axis))
        )




    final_params = (*fn.params, *out_vars, *shape_params)
    for p in final_params:
        pty = getattr(p, "type", None)
        if isinstance(pty, TensorType) and pty.storage is StorageKind.UMAT:
            raise ValueError(
                f"function {fn.name!r} parameter {p.name!r} carries unmaterialized "
                f"storage (umat); it must be materialized to a concrete residency "
                f"before lowering to TIR"
            )

    return PrimFunction(
        name=override_name if override_name is not None else fn.name,
        params=final_params,
        body=body,
        output_count=len(out_vars),
        target=target,
    )


def _build_dispatch_entry(
    group: tuple[HirFunction, ...],
    mangled_pfs: list[PrimFunction],
    target: Target,
) -> PrimFunction:
    """Build the unmangled entry PrimFunction holding the DispatchCall.

    Template params and TensorType envelope come from the first variant
    in the group; the entry forwards its own params positionally into
    each mangled callee. The body is a single ``DispatchCall`` whose
    subject is ``ShapeOf(param, axis)`` for the canonical first
    occurrence of the dispatch ``DimVar`` in the variant signature.
    """
    template = group[0]
    pat0 = template.specializations[0]
    if not isinstance(pat0, DimVarRangePat):
        raise TypeError(
            f"dispatch group {template.name!r}: only DimVarRangePat is "
            f"supported in v0"
        )
    dim_name = pat0.dim_var
    loc = locate_dim_var(template.params, dim_name)
    if loc is None:
        raise TypeError(
            f"dispatch group {template.name!r}: dispatch DimVar "
            f"{dim_name!r} not found in template signature"
        )
    param_index, axis = loc

    entry_params = tuple(
        Var(type=p.type, name=p.name) for p in template.params
    )
    is_tuple_return = isinstance(template.return_type, TupleType)
    field_types = (
        template.return_type.fields if is_tuple_return
        else (template.return_type,)
    )
    out_vars: list[Var] = []
    for i, ft in enumerate(field_types):
        flat = TensorType(
            shape=ft.shape, dtype=ft.dtype, layout=None, storage=ft.storage,
        )
        out_vars.append(
            Var(type=flat, name="out" if len(field_types) == 1 else f"out{i}")
        )
    subject_param = entry_params[param_index]
    scalar_i32 = TensorType.scalar(dtype=DType.i32, storage=StorageKind.RMEM)
    subject = ShapeOf(type=scalar_i32, param=subject_param, axis=axis)
    case_patterns: list[tuple[DimVarRangePat, ...]] = []
    case_calls: list[Evaluate] = []
    forwarded_args = (*entry_params, *out_vars)
    for variant, pf in zip(group, mangled_pfs):
        pat = variant.specializations[0]
        assert isinstance(pat, DimVarRangePat)
        case_patterns.append((pat,))





        call_args: list = list(forwarded_args)
        extra = pf.params[len(forwarded_args):]
        for extra_p in extra:

            parsed = parse_shape_var_name(extra_p.name)
            entry_p = next(
                (p for p in entry_params if parsed and p.name == parsed[0]), None
            )
            if entry_p is None:
                raise RuntimeError(
                    f"dispatch entry {template.name!r}: cannot resolve "
                    f"trailing kernel param {extra_p.name!r} on mangled "
                    f"callee {pf.name!r}"
                )
            _, ax = parsed
            call_args.append(ShapeOf(type=scalar_i32, param=entry_p, axis=ax))
        case_calls.append(symbol_call(pf, call_args))
    dispatch = DispatchCall(
        callee_name=template.name,
        subjects=(subject,),
        case_patterns=tuple(case_patterns),
        case_calls=tuple(case_calls),
        fallback=Sequential(body=(Abort(),)),
    )


    shape_param = Var(
        type=scalar_i32,
        name=shape_var_name(subject_param.name, axis),
    )
    body = Sequential(body=(dispatch, Return()))
    return PrimFunction(
        name=template.name,
        params=(*entry_params, *out_vars, shape_param),
        body=body,
        output_count=len(out_vars),
        target=target,
    )


class _MeshDeriver(ExprWalker[None]):
    """Walk a HIR expression to find meshes, keyed by topology name.

    ``cta_mesh`` / ``thread_mesh`` hold the first mesh found for each
    topology after ``visit`` returns; each stays ``None`` if its topology
    is never referenced.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cta_mesh: Mesh | None = None
        self.thread_mesh: Mesh | None = None

    def visit(self, expr):


        if self.cta_mesh is not None and self.thread_mesh is not None:
            return
        super().visit(expr)

    def visit_Call(self, call: Call) -> None:
        ty = call.type
        sl = getattr(ty, "layout", None)
        if isinstance(sl, ShardLayout):
            m = sl.mesh
            primary_name = m.topologies[0].name
            if primary_name == "cta":
                if self.cta_mesh is None:
                    self.cta_mesh = m
            elif self.thread_mesh is None:





                self.thread_mesh = m
        self.visit_operands(call)


def _derive_meshes_from_body(expr) -> tuple[Mesh | None, Mesh | None]:
    """Walk a HIR expression to find meshes, keyed by topology name.

    Returns ``(cta_mesh, thread_mesh)`` — the first mesh found for each
    topology.  Returns ``None`` for any topology not found.
    """
    deriver = _MeshDeriver()
    deriver.visit(expr)
    return deriver.cta_mesh, deriver.thread_mesh


class _CalleeCollector(ExprWalker[None]):
    """Collect the set of HIR function names called anywhere in an Expr."""

    def __init__(self) -> None:
        super().__init__()
        self.found: set[str] = set()

    def visit_Call(self, call: Call) -> None:
        tgt = call.target
        if isinstance(tgt, HirFunction):
            self.found.add(tgt.name)
        self.visit_operands(call)


def _collect_hir_callee_names(expr) -> set[str]:
    """Return the set of HIR function names called anywhere in ``expr``."""
    collector = _CalleeCollector()
    collector.visit(expr)
    return collector.found


def _topo_order_dispatch_groups(
    dispatch_view: dict[str, tuple[HirFunction, ...]],
) -> list[str]:
    """Order dispatch group names so every group appears after the groups its variants call into.

    Order dispatch group names so every group appears after the
    groups its variants call into.

    Only edges to other dispatch groups matter — calls into static
    (non-dispatch) functions don't constrain pre-lowering ordering. Source
    order is the tie-breaker so the output is deterministic.
    """
    names = list(dispatch_view.keys())
    deps: dict[str, set[str]] = {}
    for name, group in dispatch_view.items():
        d: set[str] = set()
        for variant in group:
            for callee in _collect_hir_callee_names(variant.body):
                if callee != name and callee in dispatch_view:
                    d.add(callee)
        deps[name] = d
    order: list[str] = []
    placed: set[str] = set()

    remaining = list(names)
    while remaining:
        progressed = False
        for n in list(remaining):
            if deps[n] <= placed:
                order.append(n)
                placed.add(n)
                remaining.remove(n)
                progressed = True
        if not progressed:


            order.extend(remaining)
            break
    return order


@dataclass
class HirToTirPass(ModulePass):
    """Replace every ``hir.Function`` with a ``tir.PrimFunction``.

    Meshes are auto-derived from the HIR function body (``ShardLayout.mesh``
    attributes), keyed by topology name ``"cta"`` / ``"thread"``.

    ``_cta`` / ``_thread`` are optional fallbacks for functions whose
    body does not contain mesh info (internal compat only; the public
    ``lower()`` / ``compile()`` / ``jit()`` APIs do not accept mesh kwargs).
    """

    cta_var_name: str = "block"
    thread_var_name: str = "thread"
    _cta: Mesh | None = None
    _thread: Mesh | None = None

    name: str = "hir_to_tir"
    requires: tuple[str, ...] = ()

    def run(self, module: Module) -> Module:




        try:
            target = module.resolve_target()
        except ValueError:


            target = default_target()
            module = replace(module, target=target)
        dispatch_view: dict[str, tuple[HirFunction, ...]] = {}
        for fn in module.functions:
            if isinstance(fn, HirFunction) and fn.variants:
                dispatch_view[fn.name] = fn.variants

        mangled_registry: dict[str, PrimFunction] = {}
        mangled_by_group: dict[str, list[PrimFunction]] = {}











        order = _topo_order_dispatch_groups(dispatch_view)
        for group_name in order:
            group = dispatch_view[group_name]
            lowered: list[PrimFunction] = []
            for variant in group:
                cta_mesh, thread_mesh = _derive_meshes_from_body(variant.body)
                if cta_mesh is None:
                    cta_mesh = self._cta
                if thread_mesh is None:
                    thread_mesh = self._thread
                mangled_name = _mangle_variant_name(variant)
                pf = _lower_function(
                    variant,
                    target=target,
                    cta_mesh=cta_mesh,
                    thread_mesh=thread_mesh,
                    cta_var_name=self.cta_var_name,
                    thread_var_name=self.thread_var_name,
                    override_name=mangled_name,
                    dispatch_groups=dispatch_view,
                    mangled_registry=mangled_registry,
                )
                mangled_registry[mangled_name] = pf
                lowered.append(pf)
            mangled_by_group[group_name] = lowered









        new_fns: list = []
        changed = False
        emitted_groups: set[str] = set()
        for fn in module.functions:
            if not isinstance(fn, HirFunction):
                new_fns.append(fn)
                continue
            changed = True
            group_name = fn.name
            if group_name in dispatch_view:
                if group_name in emitted_groups:
                    continue
                emitted_groups.add(group_name)
                mangled_for_group = mangled_by_group[group_name]
                new_fns.extend(mangled_for_group)
                new_fns.append(
                    _build_dispatch_entry(
                        dispatch_view[group_name], mangled_for_group, target
                    )
                )
                continue

            cta_mesh, thread_mesh = _derive_meshes_from_body(fn.body)
            if cta_mesh is None:
                cta_mesh = self._cta
            if thread_mesh is None:
                thread_mesh = self._thread
            new_fns.append(
                _lower_function(
                    fn,
                    target=target,
                    cta_mesh=cta_mesh,
                    thread_mesh=thread_mesh,
                    cta_var_name=self.cta_var_name,
                    thread_var_name=self.thread_var_name,
                    dispatch_groups=dispatch_view,
                    mangled_registry=mangled_registry,
                )
            )

        if not changed:
            return module
        new_fns = _retarget_launch_callees(new_fns)
        return replace(module, functions=tuple(new_fns))


def _retarget_launch_callees(fns: list) -> list:
    """Retarget launch callees.

    Rebuild a host launch's ``SymbolRef`` callee to the unique lowered cuda
    ``PrimFunction`` of the same name. A callee that maps to zero or several
    lowered cuda functions is an error (no guessing) — this also rejects a
    launch of a specialization group, whose variants carry mangled names.
    """
    from tilefoundry.codegen.cuda.tir.prim_function import (  # noqa: PLC0415
        _is_dispatch_entry_shape,
    )
    from tilefoundry.ir.visitor import StmtMutator  # noqa: PLC0415



    lowered_by_name: dict[str, list] = {}
    for f in fns:
        if (
            isinstance(f, PrimFunction)
            and isinstance(f.target, CudaTarget)
            and not _is_dispatch_entry_shape(f)
        ):
            lowered_by_name.setdefault(f.name, []).append(f)

    class _Rewriter(StmtMutator):
        def visit_Evaluate(self, stmt):
            if not isinstance(stmt.callable, Launch):
                return stmt
            ref = stmt.args[0]
            matches = lowered_by_name.get(ref.name, [])
            if len(matches) != 1:
                raise ValueError(
                    f"HirToTirPass: launch callee {ref.name!r} maps to "
                    f"{len(matches)} lowered cuda device kernels; expected "
                    f"exactly one (a specialization group cannot be launched)"
                )
            lowered = matches[0]
            new_ref = SymbolRef(
                name=lowered.name,
                type=callable_type_for_prim_function(lowered),
            )
            return replace(stmt, args=(new_ref, *stmt.args[1:]))

    rewriter = _Rewriter()
    return [
        rewriter.visit(f) if isinstance(f, PrimFunction) else f for f in fns
    ]


__all__ = ["HirToTirPass"]
