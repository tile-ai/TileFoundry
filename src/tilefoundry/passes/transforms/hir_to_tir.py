"""HirToTirPass — demo-path hir → tir lowering wrapped as a ModulePass.

Emits P2-shape TIR: resource introduction goes through
``LetStmt(var, value=Call(AllocTensor,
tensor_type=...), body=Sequential(...))``; tensor-pointwise HIR ops lower
to TIR effect Stmt forms (``tir.nn.ReLU(src, dst)``) anchored by
``LetStmt`` for the destination buffer.

The lowerer collects a flat sequence of "items" — either a let-binding
or an effect stmt — and folds it into a nested ``LetStmt`` chain at the
end. This keeps the collection code linear while the emitted IR is the
deep nesting the P2 spec requires.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Union

from tilefoundry.ir.core import Call, Constant, Expr, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.math.binary import Binary as HirBinary
from tilefoundry.ir.hir.math.clamp import Clamp as HirClamp
from tilefoundry.ir.hir.math.rsqrt import Rsqrt as HirRsqrt
from tilefoundry.ir.hir.math.unary import Unary as HirUnary
from tilefoundry.ir.hir.cuda.nn.mma import Mma_SM80_16x8x16 as HirMmaSM80_16x8x16
from tilefoundry.ir.hir.cuda.nn.mma import Wgmma_SM90_64x128x16 as HirWgmma_SM90
from tilefoundry.ir.hir.nn.relu import ReLU as HirReLU
from tilefoundry.ir.hir.sharding.reshard import Reshard, _shared_engine_strides
from tilefoundry.ir.hir.tensor.cache_update import CacheUpdate as HirCacheUpdate
from tilefoundry.ir.hir.tensor.cast import Cast as HirCast
from tilefoundry.ir.hir.tensor.full_like import FullLike as HirFullLike
from tilefoundry.ir.hir.tensor.gather import Gather as HirGather
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice as HirInsertSlice
from tilefoundry.ir.hir.tensor.reduce import Reduce as HirReduce
from tilefoundry.ir.hir.tensor.reshape import Reshape as HirReshape
from tilefoundry.ir.hir.tensor.tuple import Tuple
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem as HirTupleGetItem
from tilefoundry.ir.target.storage import StorageKind
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
from tilefoundry.ir.tir.cuda.nn.mma import Mma as TirMma
from tilefoundry.ir.tir.nn.relu import ReLU as TirReLU
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.reduce import Reduce as TirReduce
from tilefoundry.ir.tir.shape import ShapeOf, shape_var_name
from tilefoundry.ir.tir.sync import Sync as TirSync
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
from tilefoundry.ir.types import (
    DType,
    TensorType,
    TupleType,
    callable_type_for_prim_function,
)
from tilefoundry.ir.types.shard.layout import EMPTY_LAYOUT
from tilefoundry.ir.types.shard.layout import Layout as _Layout
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.visitor_registry.registries import (
    hir_lowering_registry,
    register_hir_lowering,
)
from tilefoundry.ir.types.shard.shard_layout import (
    ShardLayout,
    Split,
    shard_layout_local_shape,
)
from tilefoundry.ir.types.shard.shard_layout import (
    layout_axis_to_tensor_axis as _layout_axis_to_tensor_axis,
)
from tilefoundry.passes.pass_base import ModulePass


def _eval_call(op, args: tuple) -> Evaluate:
    """Place an effect-ful TIR Op in Stmt position as ``Evaluate(op, args)``."""
    return Evaluate(callable=op, args=args)


def _is_full_layout(layout) -> bool:
    """True when ``layout`` is a full (bijective) embedding — its cute
    cosize (``1 + Σ (shape[i]-1)·stride[i]``) equals the product of its
    shape — so the strides describe a complete global gather mapping
    rather than a collapsed per-instance form. Returns ``False`` when the
    strides are unavailable (unmaterialized layout), so callers fall back
    to the shared-engine path instead of crashing."""
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
    # A dynamic extent has no numeric cosize. A layout is full iff its strides
    # are the row-major (C-order) strides of its shape — equivalent to
    # ``cosize == size`` for a contiguous embedding. A dynamic dim only appears
    # as the outermost axis, so it never enters a stride product; the inner
    # extents that build the strides must be static for the check to hold.
    expected = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        nxt = shape[i + 1]
        if not isinstance(nxt, int) or not isinstance(expected[i + 1], int):
            return False
        expected[i] = expected[i + 1] * nxt
    return all(
        isinstance(strides[i], int) and strides[i] == expected[i]
        for i in range(len(shape))
    )


def _analyze_cross_warp_workspace(input_ty, reduce_axes):
    """Compute the cross-warp staging workspace requirement for a
    sharded ``Reduce``.

    Returns ``(workspace_size, dtype, lane_reduced)`` — the values the lowering
    needs to size the staging buffer. ``workspace_size`` = total non-thread mesh
    positions; ``0`` means no workspace needed. ``lane_reduced`` = whether a
    reduced Split sits on an intra-warp (lane) mesh axis: when a workspace is
    needed and ``lane_reduced`` is false, the reduction crosses warps ONLY (each
    lane keeps its own cells) and the staging buffer holds one slot per
    (warp, lane, cell). The runtime — not the lowering — derives the reduction
    tier and its ``warps_per_group`` from the operand layouts.
    """

    layout = getattr(input_ty, "layout", None)
    if not isinstance(layout, ShardLayout):
        return 0, input_ty.dtype, True

    # ``layout.layout.shape`` is the **global** cute layout shape
    # (= filled cute shape under the old convention),
    # so it can be fed straight into ``_layout_axis_to_tensor_axis``.
    cute_shape = tuple(int(s) for s in layout.layout.shape)
    pos_to_axis = _layout_axis_to_tensor_axis(cute_shape, input_ty.shape)
    rank = len(input_ty.shape)
    normalized = tuple(a % rank if a < 0 else a for a in reduce_axes)

    mesh = layout.mesh
    mesh_shape = tuple(mesh.layout.shape)
    topologies = list(mesh.topologies)

    # ── intra-warp mesh-axis range ──────────────────────────────────
    # Thread topology occupies the rightmost mesh axes (C-order
    # convention: thread is the innermost topology). Lanes within one
    # hardware warp reduce via shuffle, so only the rightmost thread
    # axes whose product stays within a single warp are intra-warp.
    # Mesh axes beyond that warp-sized suffix span multiple warps — a
    # reduced Split there needs the cross-warp workspace path, not an
    # intra-warp shuffle. (Assumes the thread topology's innermost
    # factor aligns with the warp, as the factorised meshes here do.)
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

    # Reject cross-CTA reduce explicitly so the runtime dispatch does not fall
    # back to a within-CTA tier. A "cta" topology in ``mesh.topologies`` that
    # contributes a reduced Split axis means the reduction spans CTAs and
    # requires the tier-3 ``reduce_cross_cta`` path (not yet implemented).
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
    group_count = 1  # non-reduced Split mesh extent product
    lane_reduced = False  # a reduced Split on an intra-warp (lane) axis
    for mesh_axis_idx, attr in enumerate(layout.attrs):
        if not isinstance(attr, Split):
            continue
        L = attr.axis
        if not (0 <= L < len(pos_to_axis)):
            continue
        on_reduced = pos_to_axis[L] in normalized
        # Thread axis? (rightmost thread_axes in mesh)
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

    # Total workspace slots: one per warp (all non-thread mesh positions).
    total_warps = cross_warp * group_count
    # When ``cross_warp <= 1`` no actual cross-warp summing happens —
    # each "group" is one warp — so skip the smem workspace and emit
    # the intra-warp tier-1 path (``reduce(src, dst)`` overload).
    if total_warps <= 1 or cross_warp <= 1:
        return 0, input_ty.dtype, True
    # ``lane_reduced=False`` means the reduction crosses warps ONLY — every lane
    # keeps its own independent cells. The runtime then selects the cross-warp
    # tier, whose staging buffer holds one slot per (warp, lane, cell): x32 slots
    # (sized below by the lowering).
    return total_warps, input_ty.dtype, lane_reduced


@dataclass(frozen=True)
class _Bind:
    """Fold marker: introduce a LetStmt binding `var = value`."""
    var: Var
    value: Expr


_Item = Union[_Bind, Stmt]


class _Lowerer:
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
        # Per-field Vars of a tuple-typed producer (a multi-carry
        # ``GridRegionExpr``); ``TupleGetItem`` selects a field from here.
        self._tuple_parts: dict[int, list[Var]] = {}
        # Loop-carry phi Var (id) -> its HIR init Expr, so a chain can be
        # traced back to its param-rooted gmem alias (``_param_alias_root``).
        self._carry_init: dict[int, Expr] = {}
        # Dispatch lowering context. ``dispatch_groups`` maps callee
        # name -> the overload-group tuple of HIR variants (only entries
        # with non-empty specializations). ``mangled_registry`` maps
        # ``mangled_symbol`` -> the already-lowered mangled PrimFunction
        # (filled before any caller body is lowered, so sub-call sites
        # can reference it by symbol via Evaluate(SymbolRef, args)).
        # ``caller_fn`` is the HIR function currently being lowered (used
        # to resolve caller-side ranges from its own ``specializations``).
        # ``shape_param_names`` collects ``(param_name, axis)`` pairs the
        # body references via ``ShapeOf`` so the enclosing PrimFunction
        # can add the corresponding ``<param>_shape_<axis>`` scalar
        # kernel params.
        self._dispatch_groups = dispatch_groups or {}
        self._mangled_registry = mangled_registry or {}
        self._caller_fn = caller_fn
        self._shape_param_names: set[tuple[str, int]] = (
            shape_param_names if shape_param_names is not None else set()
        )

    def _fresh(self, type_, hint: str = "v") -> Var:
        self._name_counter += 1
        return Var(type=type_, name=f"{hint}{self._name_counter}")

    def lower_expr(self, expr: Expr) -> Var:
        key = id(expr)
        if key in self._cache:
            return self._cache[key]
        if isinstance(expr, Var):
            self._cache[key] = expr
            return expr
        if isinstance(expr, Constant):
            # Lower scalar constant to a filled buffer. An unmaterialized literal
            # (storage=umat) is materialized here to register memory — the
            # backing buffer needs a concrete residency before codegen.
            const_type = expr.type
            if const_type.storage is StorageKind.UMAT:
                const_type = TensorType(
                    shape=const_type.shape,
                    dtype=const_type.dtype,
                    layout=const_type.layout,
                    storage=StorageKind.RMEM,
                )
            r = self._fresh(const_type, hint="c")
            alloc_r = Call(type=r.type, target=AllocTensorOp(tensor_type=r.type), args=())
            self._items.append(_Bind(var=r, value=alloc_r))
            self._items.append(_eval_call(Fill(), (r, expr)))
            self._cache[key] = r
            return r
        if isinstance(expr, GridRegionExpr):
            # A loop-carried region (accumulator loop) materialises its phis as
            # mutable buffers and copies the yields back each iteration; a
            # plain map region writes each body result into ``out[m, :]``.
            if expr.carried_args:
                return self._lower_grid_region_carry(expr)

            iv = expr.induction_var
            grid_ty = expr.type  # full (M, K) output type

            # Determine loop bound from GridRegion's output type
            M = grid_ty.shape[0] if grid_ty.shape else 1

            # Allocate full output tensor
            out_type = TensorType(
                shape=grid_ty.shape,
                dtype=grid_ty.dtype,
                layout=grid_ty.layout,
                storage=grid_ty.storage,
            )
            out_var = self._fresh(out_type, hint="grid_out")
            alloc_out = Call(
                type=out_type,
                target=AllocTensorOp(tensor_type=out_type),
                args=(),
            )
            self._items.append(_Bind(var=out_var, value=alloc_out))

            # Create a sub-lowerer for the body
            sub = _Lowerer()
            sub._name_counter = self._name_counter
            for k, v in self._cache.items():
                sub._cache[k] = v
            for k, v in self._tuple_parts.items():
                sub._tuple_parts[k] = v
            sub._carry_init.update(self._carry_init)
            # Bind induction var in sub-lowerer; also bind the output alloc
            sub._cache[id(iv)] = iv

            # Lower the body (per-row computation)
            body_result = sub.lower_expr(expr.body)

            # Create TensorView of output[m, :] and Copy body result into it
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
            out_view_call = Call(type=out_view_type, target=out_view_op, args=(out_var, iv))
            out_view_var = sub._fresh(out_view_type, hint="ov")
            sub._items.append(_Bind(var=out_view_var, value=out_view_call))
            sub._items.append(_eval_call(Copy(), (body_result, out_view_var)))

            self._name_counter = sub._name_counter
            body_seq = _fold_items_to_sequential(sub._items)

            # Wrap in For loop
            i32_scalar = TensorType(shape=(), dtype=DType.i32, layout=None, storage=StorageKind.RMEM)
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
        return handler(self, target, expr)

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

        # Materialise one accumulator per carry from its (lowered) initial
        # value. A fresh buffer keeps each phi mutable without aliasing the
        # init value's binding (which may be shared or a kernel param).
        acc_vars: list[Var] = []
        for init_arg in expr.init_args:
            init_var = self.lower_expr(init_arg)
            if init_var.type.storage == StorageKind.GMEM:
                # In-place carry over a gmem buffer / view: gmem is not
                # allocatable inside the kernel — bind the phi to the initial
                # binding itself. In-place ops return their dst, so the yield
                # resolves back to the same var.
                acc_vars.append(init_var)
                continue
            acc_var = self._fresh(init_var.type, hint="acc")
            alloc_acc = Call(
                type=acc_var.type,
                target=AllocTensorOp(tensor_type=acc_var.type),
                args=(),
            )
            self._items.append(_Bind(var=acc_var, value=alloc_acc))
            self._items.append(_eval_call(Copy(), (init_var, acc_var)))
            acc_vars.append(acc_var)

        # Lower the body with every phi bound to its accumulator buffer.
        sub = _Lowerer()
        sub._name_counter = self._name_counter
        for k, v in self._cache.items():
            sub._cache[k] = v
        for k, v in self._tuple_parts.items():
            sub._tuple_parts[k] = v
        sub._cache[id(iv)] = iv
        # Carry-init continuity across the sub-lowerer boundary: a chain rooted
        # in a carried var inside the body must trace to the SAME gmem alias
        # root as the carry's init (used by the reshard-owned sync).
        sub._carry_init.update(self._carry_init)
        for phi, init_arg in zip(expr.carried_args, expr.init_args):
            self._carry_init[id(phi)] = init_arg
            sub._carry_init[id(phi)] = init_arg
        for phi, acc_var in zip(expr.carried_args, acc_vars):
            sub._cache[id(phi)] = acc_var
        sub.lower_expr(expr.body)
        # All yield values are computed (SSA) before any copy-back, so a later
        # copy cannot clobber an earlier yield's inputs.
        yield_vars = [sub.lower_expr(y) for y in expr.yield_values]
        for yield_var, acc_var in zip(yield_vars, acc_vars):
            if yield_var is not acc_var:
                sub._items.append(_eval_call(Copy(), (yield_var, acc_var)))

        self._name_counter = sub._name_counter
        body_seq = _fold_items_to_sequential(sub._items)

        i32_scalar = TensorType(
            shape=(), dtype=DType.i32, layout=None, storage=StorageKind.RMEM
        )
        for_loop = For(
            induction_var=iv,
            start=Constant(value=start, type=i32_scalar),
            stop=Constant(value=stop, type=i32_scalar),
            step=Constant(value=step, type=i32_scalar),
            body=body_seq,
        )
        self._items.append(for_loop)
        # Multi-carry loops are tuple-typed; a consumer selects a field via
        # TupleGetItem, which reads ``_tuple_parts``.
        self._tuple_parts[key] = acc_vars
        self._cache[key] = acc_vars[0]
        return acc_vars[0]

    def _reshard_cross_cta_sync(self, expr) -> "Var | None":
        """The reshard-owned grid fence. A reshard whose SOURCE chain is a
        param-rooted gmem cta-shard view and whose DEST is a *different* gmem
        cta-ShardLayout re-views the ROOT buffer under new ownership — a
        cross-CTA read. The sync is intrinsic to the reshard lowering: the naive
        path lowers the producing chain, fences the grid (so every CTA's shard
        writes are visible), and returns the ROOT to reshard from; otherwise it
        returns ``None``. This is the single scenario-dispatch point (a future
        async path could skip the fence); it owns only the fence — cross-CTA data
        redistribution is not implemented here. Both the intermediate reshard
        path and the output-sink reshard path route through this helper so the
        fence is never bypassed."""
        dst_sl = expr.type.layout
        src_expr = expr.args[0]
        if (
            isinstance(dst_sl, ShardLayout)
            and expr.type.storage == StorageKind.GMEM
            and not isinstance(src_expr, Var)
            and getattr(getattr(src_expr, "type", None), "storage", None)
            == StorageKind.GMEM
            and isinstance(getattr(src_expr.type, "layout", None), ShardLayout)
            # only an actual OWNERSHIP CHANGE is a transition — an identical
            # re-view keeps the local path and never fences.
            and src_expr.type.layout != dst_sl
            and all(
                getattr(t, "name", None) == "cta"
                for t in (dst_sl.mesh.topologies or (dst_sl.mesh.topology,))
            )
        ):
            root = self._param_alias_root(src_expr)
            if root is not None:
                # Lower the producing chain first (its shard writes precede the
                # fence), fence the grid, then reshard the root param.
                self.lower_expr(src_expr)
                self._items.append(_eval_call(TirSync(mesh=dst_sl.mesh), ()))
                return root
        return None

    @register_hir_lowering(Reshard)
    def _lower_reshard(self, target, expr) -> Var:
        key = id(expr)
        src_override = self._reshard_cross_cta_sync(expr)
        src = (
            src_override
            if src_override is not None
            else self.lower_expr(expr.args[0])
        )
        src_ty = src.type
        # docs/spec/hir.md §3: the dest view layout is the
        # post-typeinfer materialized form on ``expr.type``. The
        # source-side ``TensorView`` reads ``src`` using whichever
        # stride form already materialized for ``src``:
        # - ``src.type.layout`` is a ShardLayout → reuse it
        #   verbatim (the upstream Reshard materialized it);
        # - ``src.type.layout`` is ``None`` (plain kernel-param
        #   surface) → fall back to shared-engine C-order over
        #   the dest layout's canonical shape, matching the
        #   "plain inputs are kernel-boundary shared engines"
        #   rule in spec hir §3 / function-signature binding.
        sl = expr.type.layout
        dst_shape = expr.type.shape
        view_src = src
        if isinstance(src_ty.layout, ShardLayout):
            src_sl = src_ty.layout
            # Intermediate sharded source: a ShardTensor is bound for
            # ``src``. A ``TensorView``'s memory must be a pointer (cute
            # ``make_tensor`` needs an iterator, not a ShardTensor), so
            # take ``PtrOf(src)`` first — same shape as the Reshape
            # lowering below.
            ptr_call = Call(type=src_ty, target=PtrOf(), args=(src,))
            view_src = self._fresh(src_ty, hint="ptr")
            self._items.append(_Bind(var=view_src, value=ptr_call))
        elif isinstance(sl, ShardLayout):
            # A full (bijective) fragment layout encodes its own
            # global gather permutation in the strides (e.g. mma A/B
            # fragments whose lane → (m, k) mapping is non-C-order);
            # read the source through those strides. A collapsed /
            # per-instance layout has lost the global embedding, so
            # fall back to the C-order shared engine.
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

        # TensorView(memory, shard_layout) — shard view of src
        tv = TensorView(layout=src_sl)
        tv_type = TensorType(
            shape=dst_shape,
            dtype=src_ty.dtype,
            layout=src_sl,
            storage=src_ty.storage,
        )
        tv_call = Call(type=tv_type, target=tv, args=(view_src,))
        sv = self._fresh(tv_type, hint="sv")
        self._items.append(_Bind(var=sv, value=tv_call))

        if target.storage is None:
            # without storage — pure shard view, no alloc / copy
            self._cache[key] = sv
            return sv

        # with storage — allocate plain dst, copy from shard view
        # Compute per-shard physical shape: global multi-D / mesh extents at S<> axes
        per_shard_shape = list(sl.layout.shape)
        mesh_shape = sl.mesh.layout.shape
        attr_idx = 0
        for a in sl.attrs:
            if isinstance(a, Split) and attr_idx < len(mesh_shape):
                mext = mesh_shape[attr_idx]
                if mext is None:
                    # Launch-provided (dynamic) CTA extent: this axis is the
                    # dynamic outer tile count; each CTA owns exactly one
                    # slice, so its per-shard extent is a static 1. The
                    # fixed inner tile lives on the non-split axes.
                    per_shard_shape[a.axis] = 1
                else:
                    # Post-reshard typeinfer may have rewritten the
                    # ShardLayout to reg-view local form already (Split
                    # positions size 1). Use ``max(1, ...)`` so the
                    # resulting per-shard extent is at least 1 instead
                    # of collapsing to 0 via integer division.
                    per_shard_shape[a.axis] = max(
                        1, per_shard_shape[a.axis] // mext
                    )
            attr_idx += 1
        per_shard_shape = tuple(per_shard_shape)
        # ``ShardLayout.layout`` carries the global / unsharded
        # cute shape with storage-physical strides. The TIR
        # ``var.type.shape`` remains per-shard local — that's how
        # downstream alloc / arith size their iteration.
        dst_type = TensorType(
            shape=per_shard_shape,
            dtype=src_ty.dtype,
            layout=sl,
            storage=target.storage,
        )
        dst = self._fresh(dst_type, hint="t")
        alloc = Call(
            type=dst_type,
            target=AllocTensorOp(tensor_type=dst_type),
            args=(),
        )
        self._items.append(_Bind(var=dst, value=alloc))
        self._items.append(_eval_call(Copy(), (sv, dst)))
        self._cache[key] = dst
        return dst

    # ── HIR per-shape Mma SSA value op → TIR Mma effect Stmt ──
    # HIR ``Mma_SM80_*`` / ``Wgmma_SM90_*`` are SSA value ops returning
    # a fragment tensor. Lower into ``Evaluate(tir.cuda.nn.Mma,
    # (a, b, result))`` plus an upfront ``AllocTensor`` for the
    # accumulator (matches ``HirReLU`` / ``HirMul`` lowering shape).
    @register_hir_lowering(HirMmaSM80_16x8x16)
    @register_hir_lowering(HirWgmma_SM90)
    def _lower_mma(self, target, expr) -> Var:
        key = id(expr)
        a = self.lower_expr(expr.args[0])
        b = self.lower_expr(expr.args[1])
        # Per-thread C fragment shape: for SM80 16x8x16 each lane
        # owns 4 f32 (the SM80_16x8_Row CLayout value layout (2, 2)
        # at row-major strides (1, 64)). The per-shard buffer
        # therefore stays at 4 elements in cute terms — match by
        # allocating ``shape = (2, 2)`` so the fragment has the
        # same multi-axis layout as the A/B operands after their
        # own per-shard sizing.
        if isinstance(target, HirMmaSM80_16x8x16):
            out_shape = (2, 2)
        else:
            out_shape = expr.type.shape
        out_type = TensorType(
            shape=out_shape,
            dtype=target.dtype_acc,
            layout=None,
            storage=a.type.storage,
        )
        r = self._fresh(out_type, hint="r")
        alloc_r = Call(
            type=r.type,
            target=AllocTensorOp(tensor_type=r.type),
            args=(),
        )
        self._items.append(_Bind(var=r, value=alloc_r))
        # SSA mma is value-form: ``c = a @ b``. Zero-init the
        # accumulator buffer so the underlying ``a*b + c`` PTX mma
        # produces ``a @ b`` exactly. Outer-level accumulation
        # patterns like ``add(acc, mma(a, b))`` are emitted as a
        # separate Binary stmt later in the pipeline.
        zero_const = Constant(
            value=0.0,
            type=TensorType(
                shape=(), dtype=target.dtype_acc,
                layout=EMPTY_LAYOUT, storage=None,
            ),
        )
        self._items.append(_eval_call(Fill(), (r, zero_const)))
        # TirMma operand order is (acc, lhs, rhs); the lowered path leaves
        # ``atom`` implicit (None) — the SM80 codegen handler is unchanged.
        self._items.append(_eval_call(TirMma(), (r, a, b)))
        self._cache[key] = r
        return r

    @register_hir_lowering(HirReLU)
    def _lower_relu(self, target, expr) -> Var:
        key = id(expr)
        x = self.lower_expr(expr.args[0])
        r = self._fresh(x.type, hint="r")
        # stmt-form pointwise: allocate output, then run ReLU as an
        # effect stmt that reads x and writes r.
        alloc_r = Call(
            type=r.type,
            target=AllocTensorOp(tensor_type=r.type),
            args=(),
        )
        self._items.append(_Bind(var=r, value=alloc_r))
        self._items.append(_eval_call(TirReLU(), (x, r)))
        self._cache[key] = r
        return r

    # ── pointwise binary / unary (kinded tag dispatch) ──
    @register_hir_lowering(HirBinary)
    def _lower_binary(self, target, expr) -> Var:
        key = id(expr)
        lhs = self.lower_expr(expr.args[0])
        rhs = self.lower_expr(expr.args[1])
        # Build the result type from the **lowered TIR** shapes
        # (which may already be the
        # per-shard form after a reg-storage reshard), not the
        # HIR-side ``expr.type`` (logical). Otherwise the TIR
        # verifier sees Binary(src=(1,1,1,8), dst=(1,1536)).
        out_shape = lhs.type.shape if len(lhs.type.shape) >= len(rhs.type.shape) else rhs.type.shape
        # Destination storage follows the HIR-resolved output residency
        # (order-independent), not the lowered operands — a materialized
        # constant operand must not pull the result into its register
        # buffer. An unmaterialized output is materialized at this boundary.
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
        r = self._fresh(out_type, hint="r")
        alloc_r = Call(type=r.type, target=AllocTensorOp(tensor_type=r.type), args=())
        self._items.append(_Bind(var=r, value=alloc_r))
        self._items.append(_eval_call(TirBinary(kind=target.kind), (lhs, rhs, r)))
        self._cache[key] = r
        return r

    @register_hir_lowering(HirUnary)
    def _lower_unary(self, target, expr) -> Var:
        key = id(expr)
        x = self.lower_expr(expr.args[0])
        # See HirBinary above — TIR result mirrors the lowered
        # input shape, not the HIR logical shape.
        out_type = TensorType(
            shape=x.type.shape,
            dtype=expr.type.dtype,
            layout=x.type.layout,
            storage=x.type.storage,
        )
        r = self._fresh(out_type, hint="r")
        alloc_r = Call(type=r.type, target=AllocTensorOp(tensor_type=r.type), args=())
        self._items.append(_Bind(var=r, value=alloc_r))
        self._items.append(_eval_call(TirUnary(kind=target.kind), (x, r)))
        self._cache[key] = r
        return r

    # ── pointwise unary (generic tag dispatch) ──
    @register_hir_lowering(HirRsqrt)
    def _lower_rsqrt(self, target, expr) -> Var:
        key = id(expr)
        x = self.lower_expr(expr.args[0])
        r = self._fresh(x.type, hint="r")
        alloc_r = Call(type=r.type, target=AllocTensorOp(tensor_type=r.type), args=())
        self._items.append(_Bind(var=r, value=alloc_r))
        self._items.append(_eval_call(TirUnary(kind=UnaryKind.RSQRT), (x, r)))
        self._cache[key] = r
        return r

    @register_hir_lowering(HirReshape)
    def _lower_reshape(self, target, expr) -> Var:
        key = id(expr)
        # Reshape: PtrOf(src) + TensorView with same layout,
        # new logical shape.  PtrOf gives a device pointer to
        # the per-thread buffer (no offset for reshape).
        x = self.lower_expr(expr.args[0])
        # Step 1: PtrOf(x) — take the pointer
        ptr_op = PtrOf()
        ptr_ty = x.type  # PtrOf typeinfer returns same type
        ptr_call = Call(type=ptr_ty, target=ptr_op, args=(x,))
        ptr_var = self._fresh(ptr_ty, hint="ptr")
        self._items.append(_Bind(var=ptr_var, value=ptr_call))
        # Step 2: TensorView over the ptr, with new shape
        out_ty = TensorType(
            shape=target.new_shape,
            dtype=x.type.dtype,
            layout=x.type.layout,
            storage=x.type.storage,
        )
        tv = TensorView(layout=out_ty.layout, shape=target.new_shape)
        tv_call = Call(type=out_ty, target=tv, args=(ptr_var,))
        r = self._fresh(out_ty, hint="r")
        self._items.append(_Bind(var=r, value=tv_call))
        self._cache[key] = r
        return r

    @register_hir_lowering(HirClamp)
    def _lower_clamp(self, target, expr) -> Var:
        key = id(expr)
        x = self.lower_expr(expr.args[0])
        r = self._fresh(x.type, hint="r")
        alloc_r = Call(type=r.type, target=AllocTensorOp(tensor_type=r.type), args=())
        self._items.append(_Bind(var=r, value=alloc_r))
        self._items.append(_eval_call(
            TirClamp(min_val=target.min_val, max_val=target.max_val), (x, r)))
        self._cache[key] = r
        return r

    @register_hir_lowering(HirCast)
    def _lower_cast(self, target, expr) -> Var:
        key = id(expr)
        x = self.lower_expr(expr.args[0])
        out_type = TensorType(
            shape=x.type.shape,
            dtype=target.dtype,
            layout=x.type.layout,
            storage=x.type.storage,
        )
        r = self._fresh(out_type, hint="r")
        alloc_r = Call(type=r.type, target=AllocTensorOp(tensor_type=r.type), args=())
        self._items.append(_Bind(var=r, value=alloc_r))
        self._items.append(_eval_call(TirUnary(kind=UnaryKind.CAST), (x, r)))
        self._cache[key] = r
        return r

    @register_hir_lowering(HirTupleGetItem)
    def _lower_tuple_get_item(self, target, expr) -> Var:
        key = id(expr)
        # Structural select over a tuple-typed producer (a multi-carry
        # GridRegion): lowering the producer materialises its per-field Vars in
        # ``_tuple_parts``; this just picks one.
        src = expr.args[0]
        self.lower_expr(src)
        parts = self._tuple_parts.get(id(src))
        if parts is None:
            raise TypeError(
                "hir_to_tir: TupleGetItem on a producer with no lowered tuple "
                f"fields ({type(src).__name__})"
            )
        r = parts[target.index]
        self._cache[key] = r
        return r

    @register_hir_lowering(HirFullLike)
    def _lower_full_like(self, target, expr) -> Var:
        key = id(expr)
        # Pure type-driven alloc + fill: the input expr only donates its type
        # (shape / dtype / layout / storage). Its LOWERED form is mirrored so
        # that inside a mesh scope the accumulator carries the per-shard local
        # shape (not the HIR-global one) — a global-shaped accumulator would
        # poison downstream broadcast-shape decisions.
        ty = expr.type
        storage = (
            ty.storage if ty.storage is not StorageKind.UMAT else StorageKind.RMEM
        )
        try:
            tmpl = self.lower_expr(expr.args[0])
        except Exception:
            tmpl = self._cache.get(id(expr.args[0]))
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
            # TIR var shapes are per-shard local (see the Reshard lowering); a
            # sharded type sizes its buffer / fill count from the layout.
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
        r = self._fresh(out_type, hint="c")
        alloc_r = Call(
            type=r.type, target=AllocTensorOp(tensor_type=r.type), args=()
        )
        self._items.append(_Bind(var=r, value=alloc_r))
        fill_value = Constant(
            value=target.value,
            type=TensorType(
                shape=(), dtype=ty.dtype, layout=None, storage=StorageKind.RMEM
            ),
        )
        self._items.append(_eval_call(Fill(), (r, fill_value)))
        self._cache[key] = r
        return r

    @register_hir_lowering(HirCacheUpdate)
    def _lower_cache_update(self, target, expr) -> Var:
        key = id(expr)
        cache = self.lower_expr(expr.args[0])
        self.lower_expr(expr.args[1])  # ``cur`` position (runtime row index)
        cur = self.lower_expr(expr.args[1])
        # expr.args[2] (``s``) is not consulted: v0 supports the decode-step
        # shape S_CAP == 1 (exactly one row), where s must be 1.
        new = self.lower_expr(expr.args[3])
        cache_shape = tuple(cache.type.shape)
        new_shape = tuple(expr.args[3].type.shape)
        if new_shape[1] != 1 or cache_shape[0] != 1:
            raise NotImplementedError(
                "cache_update lowering: v0 supports B == 1 and S_CAP == 1 "
                f"(single-row decode write); got cache {cache_shape}, "
                f"new {new_shape}"
            )
        # Row-slice view of the cache at runtime ``cur`` along axis 1; keep rank
        # 4 (axis-1 extent 1) so the Copy shape check matches ``new``.
        view_shape = (new_shape[0], 1, *cache_shape[2:])
        cstr = [1] * len(cache_shape)
        for i in range(len(cache_shape) - 2, -1, -1):
            cstr[i] = cstr[i + 1] * int(cache_shape[i + 1])
        view_layout = _Layout(shape=view_shape, strides=tuple(cstr))
        tv_type = TensorType(
            shape=view_shape,
            dtype=cache.type.dtype,
            layout=view_layout,
            storage=cache.type.storage,
        )
        tv_call = Call(type=tv_type, target=TensorView(layout=view_layout), args=(cache, cur))
        sv = self._fresh(tv_type, hint="sv")
        self._items.append(_Bind(var=sv, value=tv_call))
        # In-place per the op contract (anchored on the cache buffer): write the
        # new row into the cache and return the cache var itself.
        self._items.append(_eval_call(Copy(), (new, sv)))
        self._cache[key] = cache
        return cache

    @register_hir_lowering(HirInsertSlice)
    def _lower_insert_slice(self, target, expr) -> Var:
        key = id(expr)
        dst = self.lower_expr(expr.args[0])
        upd = self.lower_expr(expr.args[1])
        # 1-D window (insert_slice is rank-1). The window starts at the scalar
        # offset; it is a tile of the update's own extent, so the offset is the
        # coord. Write ``update`` into that window IN PLACE and return ``dst`` —
        # a loop-carried ``ov = insert_slice(ov, …)`` therefore reuses the single
        # carried buffer with no replacement allocation, and the yield aliases
        # the carry buffer.
        coord = self._insert_slice_coord(expr.args[2])
        upd_shape = tuple(upd.type.shape)
        # The window carries the update's layout: for a sharded update this is a
        # ShardLayout, so the Copy verifier sees matching ShardLayouts and the
        # emitter derives the per-shard tile size (``local()`` coalesces the dst
        # to a flat 1-D view, so K must be the whole window, not just axis 0).
        if isinstance(upd.type.layout, ShardLayout):
            win_layout = upd.type.layout
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
            type=win_type, target=TensorView(layout=win_layout), args=(dst, coord)
        )
        win = self._fresh(win_type, hint="isv")
        self._items.append(_Bind(var=win, value=win_call))
        self._items.append(_eval_call(Copy(), (upd, win)))
        self._cache[key] = dst
        return dst

    def _param_alias_root(self, expr, _depth: int = 0):
        """Walk Reshard / loop-carry chains to the underlying kernel-param Var
        (the gmem alias root), or ``None``. Used by the cross-CTA
        sync-then-reshard rule in ``_lower_reshard``."""
        if _depth > 64:
            return None
        if isinstance(expr, Var):
            if id(expr) in self._carry_init:
                return self._param_alias_root(
                    self._carry_init[id(expr)], _depth + 1
                )
            return expr if self._cache.get(id(expr)) is expr else None
        if isinstance(expr, GridRegionExpr):
            for a in expr.init_args:
                r = self._param_alias_root(a, _depth + 1)
                if r is not None:
                    return r
            return None
        if isinstance(getattr(expr, "target", None), Reshard):
            return self._param_alias_root(expr.args[0], _depth + 1)
        return None

    def _insert_slice_coord(self, off_expr):
        """The scalar window index for an in-place ``insert_slice``: the window
        is one tile of the update's own extent starting at the offset, so the
        offset is the tile index (matching the ``local_tile`` coord). A
        compile-time offset folds to a ``Constant`` scalar (emitted as a
        literal coordinate); a runtime scalar offset lowers to its scalar Var
        (its single element is read at the coordinate site)."""
        i32 = TensorType(
            shape=(), dtype=DType.i32, layout=None, storage=StorageKind.RMEM
        )
        if isinstance(off_expr, Constant):
            val = off_expr.value
            elem = int(val[0]) if isinstance(val, (list, tuple)) else int(val)
            return Constant(value=elem, type=i32)
        # A runtime offset lowers to its scalar Var (a native scalar index, e.g.
        # the loop induction variable).
        return self.lower_expr(off_expr)

    # ── reduce (generic tag dispatch) ──
    @register_hir_lowering(HirGather)
    def _lower_gather(self, target, expr) -> Var:
        key = id(expr)
        x = self.lower_expr(expr.args[0])
        idx = self.lower_expr(expr.args[1])
        # Compute view layout from sliced shape
        view_shape = expr.type.shape
        view_layout = TensorView.layout_for_slice(
            src_shape=tuple(x.type.shape),
            axis=target.axis,
            sliced_shape=view_shape,
        )
        tv = TensorView(layout=view_layout)
        tv_type = TensorType(
            shape=view_shape,
            dtype=expr.type.dtype,
            layout=view_layout,
            storage=x.type.storage,
        )
        tv_call = Call(type=tv_type, target=tv, args=(x, idx))
        sv = self._fresh(tv_type, hint="sv")
        self._items.append(_Bind(var=sv, value=tv_call))
        self._cache[key] = sv
        return sv

    @register_hir_lowering(HirReduce)
    def _lower_reduce(self, target, expr) -> Var:
        key = id(expr)
        x = self.lower_expr(expr.args[0])
        axes = target.axes
        # The HIR Reduce typeinfer already computed the reduced
        # ``ShardLayout`` (cute dims on reduced axes collapsed to
        # size 1 / stride 0; corresponding Split attrs rewritten to
        # Broadcast).  Reuse that layout instead of copying the
        # input layout — otherwise the TIR output type carries an
        # un-reduced cute shape and downstream codegen iterates
        # the wrong per-thread count.
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
        r = self._fresh(out_type, hint="r")
        alloc_r = Call(type=r.type, target=AllocTensorOp(tensor_type=r.type), args=())
        self._items.append(_Bind(var=r, value=alloc_r))

        # Analyse the input ShardLayout for cross-warp staging.
        # A reduce axis covered by a Split
        # on a non-``thread`` topology (i.e. ``warp`` / ``cta`` /
        # ``cluster`` / ...) cannot be folded by ``__shfl_xor_sync``
        # alone; the runtime needs a small staging buffer the
        # warps cooperatively write into and read back. The buffer
        # size is the product of the cross-warp mesh axis extents.
        tir_args: tuple = (x, r)
        # The cross-warp analysis must run on the *HIR* input
        # type — its ``shape`` matches
        # the user-facing axes the HIR Reduce was authored against
        # (logical ``(1, 1536)``). After lowering, ``x.type.shape``
        # may be the per-shard local form ``(1, 1, 1, 8)``; using
        # that would silently mis-map ``axes=(-1,)`` to a non-Split
        # cute position and skip the workspace alloc.
        ws_size, ws_dtype, lane_reduced = _analyze_cross_warp_workspace(
            expr.args[0].type, axes
        )
        if ws_size > 0:
            # The runtime stages one warp-partial PER OUTPUT CELL: scale the
            # workspace by the output's per-thread cell count (1 for scalar
            # reduces — the historical size).
            n_cells = 1
            for dim in r.type.shape:
                if isinstance(dim, int):
                    n_cells *= dim
            ws_size *= max(1, n_cells)
            if not lane_reduced:
                # cross-warp-only: per (warp, lane, cell) staging slots.
                ws_size *= 32
            ws_type = TensorType(
                shape=(ws_size,),
                dtype=ws_dtype,
                layout=None,
                storage=StorageKind.SMEM,
            )
            ws = self._fresh(ws_type, hint="ws")
            alloc_ws = Call(
                type=ws.type,
                target=AllocTensorOp(tensor_type=ws.type),
                args=(),
            )
            self._items.append(_Bind(var=ws, value=alloc_ws))
            tir_args = (x, r, ws)

        reduce_op = TirReduce(axes=axes, kind=target.kind)
        self._items.append(_eval_call(reduce_op, tir_args))
        self._cache[key] = r
        return r

    @register_hir_lowering(HirFunction)
    def _lower_hir_function(self, target, expr) -> Var:
        return self._lower_hir_call(expr, target)

    # ── sub-call dispatch ────────────────────────────────────────────────
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
        # Lower the call args first so they have TIR Vars.
        arg_vars: list[Var] = [self.lower_expr(a) for a in call.args]
        # Allocate output buffer matching call.type.
        out_type = call.type
        out_var = self._fresh(out_type, hint="cr")
        alloc_out = Call(
            type=out_type,
            target=AllocTensorOp(tensor_type=out_type),
            args=(),
        )
        self._items.append(_Bind(var=out_var, value=alloc_out))
        # Resolve the dispatch DimVar from the callee group (per the
        # canonical first-occurrence rule). All variants in a v0 group
        # share the same dispatch DimVar by construction (the validator
        # in script.py mandates one DimVarRangePat per variant; the
        # group's overload is over the same name).
        first_variant = group[0]
        first_pat = first_variant.specializations[0]
        if not isinstance(first_pat, DimVarRangePat):
            raise TypeError(
                f"HIR Call to {callee_hir.name!r}: only DimVarRangePat "
                f"dispatch is supported in v0"
            )
        dim_name = first_pat.dim_var
        loc = _locate_dim_var_in_params(first_variant.params, dim_name)
        if loc is None:
            raise TypeError(
                f"HIR Call to {callee_hir.name!r}: dispatch DimVar "
                f"{dim_name!r} not found in callee signature"
            )
        param_index, axis = loc
        # Caller-side range from the call argument's shape entry at the
        # callee's canonical (param_index, axis).
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
        # Reachable variants in source order.
        reachable: list[tuple[HirFunction, DimVarRangePat]] = []
        for variant in group:
            pat = variant.specializations[0]
            if not isinstance(pat, DimVarRangePat):
                continue
            # Half-open intersection: [pat.lo, pat.hi) ∩ [c_lo, c_hi) != empty.
            if pat.lo < c_hi and c_lo < pat.hi:
                reachable.append((variant, pat))
        if not reachable:
            raise TypeError(
                f"HIR Call to {callee_hir.name!r}: empty reachable "
                f"specialization set; caller range [{c_lo}, {c_hi}) "
                f"does not intersect any callee specialization range"
            )
        # The subject ShapeOf reads the caller's positional argument Var.
        # For a non-Var arg the lowering already produced a Var via
        # ``arg_vars`` above; the argument's Var is the right subject
        # because the runtime shape is the dispatch axis on that Var.
        subject_param = arg_vars[param_index]
        # ShapeOf.param must be a param of the enclosing PrimFunction
        # (verifier rule). If the resolved Var is one of the caller's
        # params, use it directly; otherwise the body produced a fresh
        # Var and there is no kernel scalar — this is rejected by the
        # verifier downstream, so surface a clearer error here.
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
        scalar_i32 = TensorType.scalar(dtype=DType.i32)
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
            # call args = caller-arg TIR vars + out_var, then
            # forward any trailing <param>_shape_<axis> kernel scalars
            # the mangled callee declared (it grew them itself because
            # its own body lowers nested ShapeOf / DispatchCall sites).
            # Map each trailing scalar back to the caller-arg position
            # by matching <base_name> against the callee's user-param
            # names, then emit ShapeOf(<caller-arg Var>, axis) and
            # record the corresponding <caller-arg name>_shape_<axis>
            # on the enclosing PrimFunction so it grows the matching
            # trailing kernel param.
            call_args: list = [*arg_vars, out_var]
            head_count = len(variant.params) + mangled_pf.output_count
            trailing = mangled_pf.params[head_count:]
            for tp in trailing:
                base_name, axis_int = _parse_shape_param_name(
                    tp.name, callee_hir.name, mangled_pf.name,
                )
                user_idx = _find_user_param_index(
                    variant.params, base_name, callee_hir.name, mangled_pf.name,
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


def _parse_shape_param_name(
    name: str, callee_name: str, mangled_name: str
) -> tuple[str, int]:
    """Split ``<base>_shape_<axis>`` into ``(base, axis)``.

    Used to map a mangled callee's trailing kernel-scalar param back to
    its originating user-param name and tensor axis.
    """
    marker = "_shape_"
    idx = name.rfind(marker)
    if idx < 0:
        raise RuntimeError(
            f"HIR Call to {callee_name!r}: mangled callee {mangled_name!r} "
            f"has trailing param {name!r} that is not a "
            f"<base>_shape_<axis> shape scalar"
        )
    base = name[:idx]
    try:
        axis = int(name[idx + len(marker):])
    except ValueError as e:
        raise RuntimeError(
            f"HIR Call to {callee_name!r}: mangled callee {mangled_name!r} "
            f"has trailing param {name!r} with non-integer axis suffix"
        ) from e
    return base, axis


def _find_user_param_index(
    params: tuple[Var, ...], base_name: str,
    callee_name: str, mangled_name: str,
) -> int:
    for i, p in enumerate(params):
        if p.name == base_name:
            return i
    raise RuntimeError(
        f"HIR Call to {callee_name!r}: mangled callee {mangled_name!r} "
        f"trailing scalar references user param {base_name!r} that is "
        f"not in the callee user-param list"
    )


def _locate_dim_var_in_params(
    params: tuple[Var, ...], dim_var_name: str
) -> tuple[int, int] | None:
    """First (param_index, axis) where a named DimVar appears in params.

    Canonical scan order is (param_index ascending, axis ascending).
    """
    for i, p in enumerate(params):
        shape = getattr(p.type, "shape", None)
        if shape is None:
            continue
        for axis, dim in enumerate(shape):
            if getattr(dim, "name", None) == dim_var_name:
                return (i, axis)
    return None


def _fold_items_to_sequential(items: list[_Item]) -> Sequential:
    """Turn a flat item list into a nested ``LetStmt`` chain wrapped in a
    ``Sequential``."""
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
        # A cross-CTA ownership-change reshard fences the grid before the copy —
        # the same reshard-owned sync as the intermediate path, so an output-sink
        # reshard never bypasses the fence. The output still copies directly into
        # ``out`` (no extra sharded temporary), reading the synced root.
        override = lo._reshard_cross_cta_sync(body_expr)
        inner = override if override is not None else lo.lower_expr(body_expr.args[0])
        inner_ty = inner.type
        # Use the post-typeinfer materialized layout from ``body_expr.type``
        # — ``body_expr.target.layout``
        # may still carry the sugar ``strides=None`` form that typeinfer
        # discharges later.
        sl = getattr(body_expr.type, "layout", None)
        if sl is None:
            sl = body_expr.target.layout
        if sl is None:
            sl = getattr(inner_ty, "layout", None)
        # ``sl`` already carries the global cute shape from the
        # reshard / parser layer; no re-expansion from local needed
        # any more.
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
    cta_mesh: Mesh | None,
    thread_mesh: Mesh | None,
    cta_var_name: str = "block",
    thread_var_name: str = "thread",
    out_var_name: str = "out",
    override_name: str | None = None,
    dispatch_groups: "dict[str, tuple[HirFunction, ...]] | None" = None,
    mangled_registry: "dict[str, PrimFunction] | None" = None,
) -> PrimFunction:
    """Materialise the HIR `Function(params) -> tensor` as an
    explicit-output-param ``PrimFunction``. The function-end
    sink — the outermost Reshard write to global — is rewritten as a
    ``Copy(<reg/shared result>, out)`` into the new ``out`` parameter
    instead of allocating a fresh global tensor.
    """
    # The TIR-level output Var carries a plain (non-Shard)
    # ``TensorType`` because the kernel param is a
    # raw global pointer; any shard wrap is added explicitly by the
    # body via a ``TensorView`` LetStmt. Inheriting a ShardLayout
    # from ``fn.return_type`` would cause the kernel-param wrapper
    # to ``make_shard_tensor(...)`` automatically, double-wrapping
    # the subsequent body-side ``sv*`` view.

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
    # Seed the shape-scalar plumbing with every (param, axis) whose
    # type entry is a ``DimVar``. The kernel body needs the runtime
    # extent to size loop counts (binary / fill / copy) when the buffer
    # shape is dynamic; the dispatch entry forwards these scalars to
    # the matching variant via the trailing-shape mechanism in
    # ``_build_dispatch_entry``.
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
    # wrap each present mesh as a MeshScope; skip ones that aren't

    inner_seq = _fold_items_to_sequential(lo._items)
    # wrap each present mesh as a MeshScope; skip ones that aren't
    # used. ``_lower_function`` no longer fabricates mesh structure beyond
    # what ``ShardLayout`` types in the body actually reference. A function
    # with no mesh information at all lowers to a bare body sequence.
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
        # CTA absent but thread present — keep the thread MeshScope as
        # the outermost wrapper.
        body = Sequential(body=(scoped, Return()))
    else:
        # No mesh at all — body is just the lowered sequential plus return.
        body = Sequential(body=(*inner_seq.body, Return()))

    # Append <param>_shape_<axis>: i32 kernel scalar params for any
    # ShapeOf referenced by the body. Place them after the buffer params
    # (between input + output buffer params and any future scalars).
    shape_params: list[Var] = []
    scalar_i32_ty = TensorType.scalar(dtype=DType.i32)
    for pname, axis in sorted(shape_param_names):
        shape_params.append(
            Var(type=scalar_i32_ty, name=shape_var_name(pname, axis))
        )

    # An unmaterialized value must be materialized to a concrete residency
    # before it reaches TIR; a kernel param / output carrying umat would have
    # no memory space for the launch ABI, placement, or copies.
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
    )


def _build_dispatch_entry(
    group: tuple[HirFunction, ...],
    mangled_pfs: list[PrimFunction],
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
    loc = _locate_dim_var_in_params(template.params, dim_name)
    if loc is None:
        raise TypeError(
            f"dispatch group {template.name!r}: dispatch DimVar "
            f"{dim_name!r} not found in template signature"
        )
    param_index, axis = loc
    # Fresh entry params + output buffer params with template shapes.
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
    scalar_i32 = TensorType.scalar(dtype=DType.i32)
    subject = ShapeOf(type=scalar_i32, param=subject_param, axis=axis)
    case_patterns: list[tuple[DimVarRangePat, ...]] = []
    case_calls: list[Evaluate] = []
    forwarded_args = (*entry_params, *out_vars)
    for variant, pf in zip(group, mangled_pfs):
        pat = variant.specializations[0]
        assert isinstance(pat, DimVarRangePat)
        case_patterns.append((pat,))
        # Forward exactly the params the mangled callee expects: the
        # mangled callee's leading params are (input buffers, output
        # buffers) by construction; any trailing shape-scalar params
        # the callee declared must be matched by re-emitting the same
        # ShapeOf expressions here.
        call_args: list = list(forwarded_args)
        extra = pf.params[len(forwarded_args):]
        for extra_p in extra:
            # Match "<param>_shape_<axis>" name back to (param, axis).
            ep_name = extra_p.name
            for entry_p in entry_params:
                prefix = entry_p.name + "_shape_"
                if ep_name.startswith(prefix):
                    try:
                        ax = int(ep_name[len(prefix):])
                    except ValueError:
                        continue
                    call_args.append(
                        ShapeOf(type=scalar_i32, param=entry_p, axis=ax)
                    )
                    break
            else:
                raise RuntimeError(
                    f"dispatch entry {template.name!r}: cannot resolve "
                    f"trailing kernel param {ep_name!r} on mangled "
                    f"callee {pf.name!r}"
                )
        case_calls.append(symbol_call(pf, call_args))
    dispatch = DispatchCall(
        callee_name=template.name,
        subjects=(subject,),
        case_patterns=tuple(case_patterns),
        case_calls=tuple(case_calls),
        fallback=Sequential(body=(Abort(),)),
    )
    # Entry's <param>_shape_<axis>: i32 kernel param for the dispatch
    # axis. (Only one needed in v0 — the single canonical subject.)
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
    )


def _derive_meshes_from_body(expr, topologies: tuple) -> tuple[Mesh | None, Mesh | None]:
    """Walk a HIR expression to find meshes, keyed by topology name.

    Returns ``(cta_mesh, thread_mesh)`` — the first mesh found for each
    topology.  Returns ``None`` for any topology not found.
    """

    cta_mesh: Mesh | None = None
    thread_mesh: Mesh | None = None

    def walk(e):
        nonlocal cta_mesh, thread_mesh
        if cta_mesh is not None and thread_mesh is not None:
            return
        if isinstance(e, Tuple):
            for elt in e.elements:
                walk(elt)
            return
        if isinstance(e, Call):
            ty = e.type
            sl = getattr(ty, "layout", None)
            if isinstance(sl, ShardLayout):
                m = sl.mesh
                # ``mesh.topology`` may be a ``Topology`` dataclass or a
                # plain string depending on how the Mesh was authored
                # (sugar paths pass topology names as strings).
                primary = m.topology
                primary_name = primary.name if hasattr(primary, "name") else str(primary)
                if primary_name == "cta":
                    if cta_mesh is None:
                        cta_mesh = m
                elif thread_mesh is None:
                    # Any non-cta primary (``thread`` / ``warp`` /
                    # ``lane`` / ...) lives
                    # inside one CTA — wrap as the inner ``thread_mesh``
                    # so the verifier's MeshScope check passes and
                    # codegen still threads the right Topology<scope>
                    # all the way through.
                    thread_mesh = m
            for a in e.args:
                walk(a)

    walk(expr)
    return cta_mesh, thread_mesh


def _collect_hir_callee_names(expr) -> set[str]:
    """Return the set of HIR function names called anywhere in ``expr``."""
    found: set[str] = set()

    def walk(e) -> None:
        if isinstance(e, Call):
            tgt = e.target
            if isinstance(tgt, HirFunction):
                found.add(tgt.name)
            for a in e.args:
                walk(a)
            return
        if isinstance(e, GridRegionExpr):
            walk(e.body)
            return
        if isinstance(e, Tuple):
            for elt in e.elements:
                walk(elt)

    walk(expr)
    return found


def _topo_order_dispatch_groups(
    dispatch_view: dict[str, tuple[HirFunction, ...]],
) -> list[str]:
    """Order dispatch group names so every group appears after the
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
    # Iterative Kahn-style walk in source order.
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
            # Cycle among dispatch groups — give up topo and fall back
            # to source order so we still produce a deterministic result.
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
        # A dispatch prototype (an HIR Function carrying ``variants``) lowers
        # through the dispatch path: ``dispatch_view`` maps its name to its
        # variant tuple. A normal function (no variants, real body) lowers on
        # the single-body path; non-HIR functions pass through unchanged.
        dispatch_view: dict[str, tuple[HirFunction, ...]] = {}
        for fn in module.functions:
            if isinstance(fn, HirFunction) and fn.variants:
                dispatch_view[fn.name] = fn.variants

        mangled_registry: dict[str, PrimFunction] = {}
        mangled_by_group: dict[str, list[PrimFunction]] = {}

        # ── pre-lower every variant body to its mangled PrimFunction.
        #
        # A variant body may call into another dispatch group whose
        # mangled callees must already be in ``mangled_registry`` at the
        # time the caller body is lowered. To make the pass independent
        # of ``Module.functions`` ordering, lower groups in a topological
        # order driven by the inter-group HIR-call graph (callee group
        # before caller group). Static (non-dispatch) functions are
        # deferred — their bodies may also reference dispatch
        # groups, but by then the registry is complete.
        order = _topo_order_dispatch_groups(dispatch_view)
        for group_name in order:
            group = dispatch_view[group_name]
            lowered: list[PrimFunction] = []
            for variant in group:
                cta_mesh, thread_mesh = _derive_meshes_from_body(
                    variant.body, variant.topologies
                )
                if cta_mesh is None:
                    cta_mesh = self._cta
                if thread_mesh is None:
                    thread_mesh = self._thread
                mangled_name = _mangle_variant_name(variant)
                pf = _lower_function(
                    variant,
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

        # ── emit final function list in source order.
        #
        # For each HIR function in ``module.functions``:
        #   - dispatch group: emit its mangled variants (already lowered)
        #     then its unmangled entry PrimFunction;
        #   - static function: lower its body now (registry is complete,
        #     so any sub-call into a dispatch group resolves cleanly);
        #   - non-HIR function: pass through.
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
                        dispatch_view[group_name], mangled_for_group
                    )
                )
                continue
            # Static single-body path.
            cta_mesh, thread_mesh = _derive_meshes_from_body(
                fn.body, fn.topologies
            )
            if cta_mesh is None:
                cta_mesh = self._cta
            if thread_mesh is None:
                thread_mesh = self._thread
            new_fns.append(
                _lower_function(
                    fn,
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
    """Rebuild a host launch's ``SymbolRef`` callee to the unique lowered cuda
    ``PrimFunction`` of the same name. A callee that maps to zero or several
    lowered cuda functions is an error (no guessing) — this also rejects a
    launch of a specialization group, whose variants carry mangled names."""
    from tilefoundry.codegen.cuda.tir.prim_function import (  # noqa: PLC0415
        _is_dispatch_entry_shape,
    )
    from tilefoundry.ir.visitor import StmtMutator  # noqa: PLC0415

    # Only single-body device kernels are retarget targets; a dispatch entry is
    # host-only (no device kernel) so it must not be matched.
    lowered_by_name: dict[str, list] = {}
    for f in fns:
        if (
            isinstance(f, PrimFunction)
            and f.target.name == "cuda"
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
