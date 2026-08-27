"""Verify TIR function structure, bindings, calls, and contextual placement.

The checked rule set is defined by
[tir §1.3](docs/spec/tir.md#13-primfunction).
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

from tilefoundry.ir.core import Expr, TypeInferContext, Var, VerifyError
from tilefoundry.ir.core.expr import Call, Constant
from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.hir.function import (
    Function as HirFunction,
)
from tilefoundry.ir.hir.verify import verify_function
from tilefoundry.ir.types import (
    DType,
    TensorType,
    UnitType,
    callable_type_for_prim_function,
)
from tilefoundry.ir.types.dim import (
    DimAdd,
    DimFloorDiv,
    DimMax,
    DimMin,
    DimMod,
    DimMul,
    DimSub,
)
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.shard_layout import ShardLayout
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.visitor import ExprVisitor
from tilefoundry.target import CudaTarget
from tilefoundry.utils.spec_ref import spec_ref_render
from tilefoundry.visitor_registry import verify_stmt_registry

from .dispatch import DispatchCall
from .launch import Launch
from .memory import AllocTensor as AllocTensorOp
from .prim_function import PrimFunction
from .shape import ShapeOf, is_hidden_shape_scalar, is_shape_scalar, parse_shape_var_name
from .stmts import (
    Abort,
    Evaluate,
    For,
    If,
    LetStmt,
    MeshScope,
    Return,
    Sequential,
    While,
)
from .symbol_ref import SymbolRef

_PRIM_FUNCTION = "[tir §1.3](docs/spec/tir.md#13-primfunction)"


def verify_prim_function(fn: PrimFunction, *, module_fns: Iterable[PrimFunction] = ()) -> None:
    """Per [tir §1.3](docs/spec/tir.md#13-primfunction)'s rule list."""
    _check_param_homogeneity(fn)
    ctx = TypeInferContext()
    scope: list[Mesh] = []

    module_fn_map: dict[str, tuple[PrimFunction, ...]] = {}
    for f in module_fns:
        module_fn_map[f.name] = (*module_fn_map.get(f.name, ()), f)

    bound_var_ids: set[int] = {id(p) for p in fn.params}
    _walk_stmt(fn.body, ctx, scope, fn, module_fn_map, bound_var_ids)


def _check_param_homogeneity(fn: PrimFunction) -> None:
    """Param homogeneity ([tir §1.3](docs/spec/tir.md#13-primfunction)).

    Param homogeneity ([tir §1.3](docs/spec/tir.md#13-primfunction)): all params' layouts must be
    consistently ShardLayout or all non-ShardLayout. Mixed is error.
    """
    kinds = []
    for p in fn.params:
        ty = p.type
        if not isinstance(ty, TensorType):
            continue
        kinds.append(isinstance(ty.layout, ShardLayout))
    if kinds and len(set(kinds)) > 1:
        raise VerifyError(
            f"PrimFunction {fn.name!r}: {spec_ref_render(_PRIM_FUNCTION)} param layouts "
            f"must be homogeneously ShardLayout or non-ShardLayout, got mixed"
        )


def _walk_stmt(stmt, ctx, scope, fn, module_fn_map, bound_var_ids: set[int]):
    match stmt:
        case Sequential():
            for s in stmt.body:
                _walk_stmt(s, ctx, scope, fn, module_fn_map, bound_var_ids)
            return
        case LetStmt():
            if id(stmt.var) in bound_var_ids:
                raise VerifyError(
                    f"LetStmt binding {stmt.var.name!r}: "
                    f"{spec_ref_render(_PRIM_FUNCTION)} var must be a fresh Var; this Var "
                    f"is already bound in an outer scope"
                )

            value_ty = ctx.type_of(stmt.value)
            if stmt.var.type != value_ty:
                raise VerifyError(
                    f"LetStmt binding {stmt.var.name!r}: var.type {stmt.var.type} "
                    f"!= value.type {value_ty}"
                )

            _reject_nested_alloc_tensor(stmt.value, at_letstmt_value=True)
            _check_embedded_sharding(stmt.value, scope, fn)

            bound_var_ids.add(id(stmt.var))
            _walk_stmt(stmt.body, ctx, scope, fn, module_fn_map, bound_var_ids)
            return
        case For():
            _check_rank0_int(ctx, stmt, stmt.start, "For.start")
            _check_rank0_int(ctx, stmt, stmt.stop, "For.stop")
            _check_rank0_int(ctx, stmt, stmt.step, "For.step")
            if isinstance(stmt.step, Constant) and stmt.step.value == 0:
                raise VerifyError("For.step must not be 0")
            iv_ty = stmt.induction_var.type
            if not (
                isinstance(iv_ty, TensorType)
                and iv_ty.shape == ()
                and iv_ty.dtype in (DType.i32, DType.i64)
            ):
                raise VerifyError("For.induction_var must be rank-0 integer")
            _walk_stmt(stmt.body, ctx, scope, fn, module_fn_map, bound_var_ids)
            return
        case While():
            _check_rank0_bool(ctx, stmt, stmt.cond)
            _walk_stmt(stmt.body, ctx, scope, fn, module_fn_map, bound_var_ids)
            return
        case If():
            _check_rank0_bool(ctx, stmt, stmt.cond)
            _walk_stmt(stmt.then_body, ctx, scope, fn, module_fn_map, bound_var_ids)
            _walk_stmt(stmt.else_body, ctx, scope, fn, module_fn_map, bound_var_ids)
            return
        case MeshScope():
            scope.append(stmt.mesh)
            _walk_stmt(stmt.body, ctx, scope, fn, module_fn_map, bound_var_ids)
            scope.pop()
            return
        case Return() | Abort():
            return
        case DispatchCall():
            _verify_dispatch_call(stmt, fn, module_fn_map, ctx)
            return
        case Evaluate():
            if isinstance(stmt.callable, Launch):
                _verify_launch(stmt, fn, module_fn_map, ctx)
            elif isinstance(stmt.callable, SymbolRef):
                _verify_symbol_call(stmt, fn, module_fn_map, ctx)
            else:
                op = stmt.callable
                op_cls = type(op)
                fn_verify = verify_stmt_registry.lookup(op_cls)
                if fn_verify is None:
                    raise VerifyError(f"no verify_stmt registered for Op {op_cls.__name__}")

                ctx.mesh_scope = tuple(scope)
                call = Call(type=UnitType(), target=op, args=stmt.args)
                fn_verify(call, ctx)

            for arg in stmt.args:
                _reject_nested_alloc_tensor(arg, at_letstmt_value=False)
                _check_embedded_sharding(arg, scope, fn)
            return

    fn_verify = verify_stmt_registry.lookup(type(stmt))
    if fn_verify is None:
        raise VerifyError(f"no verify_stmt registered for {type(stmt).__name__}")
    fn_verify(stmt, ctx)
    for field_name, field_val in _iter_stmt_expr_fields(stmt):
        _reject_nested_alloc_tensor(field_val, at_letstmt_value=False)
        _check_embedded_sharding(field_val, scope, fn)


def _iter_stmt_expr_fields(stmt):
    if not dataclasses.is_dataclass(stmt):
        return
    for f in dataclasses.fields(stmt):
        val = getattr(stmt, f.name)
        if isinstance(val, Expr):
            yield f.name, val


def _check_rank0_int(ctx, stmt, expr, field: str):
    t = ctx.type_of(expr)
    if not (isinstance(t, TensorType) and t.shape == () and t.dtype in (DType.i32, DType.i64)):
        raise VerifyError(f"{field} must be rank-0 integer, got {t}")


def _check_rank0_bool(ctx, stmt, expr):
    t = ctx.type_of(expr)
    if not (isinstance(t, TensorType) and t.shape == () and t.dtype == DType.bool):
        raise VerifyError(f"condition must be rank-0 bool, got {t}")


class _AllocTensorRejectingVisitor(ExprVisitor[None]):
    def __init__(self, *, at_letstmt_value: bool) -> None:
        super().__init__()
        self.at_letstmt_value = at_letstmt_value

    def visit_Call(self, expr: Call, ctx=None) -> None:
        if isinstance(expr.target, AllocTensorOp):
            if not self.at_letstmt_value:
                raise VerifyError(
                    f"{spec_ref_render(_PRIM_FUNCTION)}: Call(AllocTensor, ...) may only appear "
                    f"as a direct LetStmt.value; found nested inside another Expr"
                )
            self.at_letstmt_value = False
            self.visit_operands(expr, ctx)
            return
        self.at_letstmt_value = False
        self.visit_operands(expr, ctx)

    def default_visit(self, expr, ctx=None) -> None:
        return None


def _reject_nested_alloc_tensor(expr: Expr, *, at_letstmt_value: bool) -> None:
    """Reject an ``AllocTensor`` call nested outside a ``LetStmt`` value.

    ``Call(AllocTensor, ...)`` may appear only as ``LetStmt.value``.

    See [tir §1.3](docs/spec/tir.md#13-primfunction).
    """
    visitor = _AllocTensorRejectingVisitor(at_letstmt_value=at_letstmt_value)
    visitor.visit(expr)


def _check_embedded_sharding(expr: Expr, scope, fn):
    """Require embedded shard layouts to use a scoped or parameter mesh.

    See [tir §1.3](docs/spec/tir.md#13-primfunction).
    """
    to_visit = [expr]
    while to_visit:
        e = to_visit.pop()
        if isinstance(e, Call):
            op = e.target
            for attr_name, attr_val in _iter_op_attrs(op):
                if isinstance(attr_val, ShardLayout):
                    _assert_mesh_in_scope(attr_val.mesh, scope, fn)
            to_visit.extend(e.args)


def _iter_op_attrs(op):
    for info in type(op).params():
        if info.kind == "attribute":
            yield info.name, getattr(op, info.name, None)


def _assert_mesh_in_scope(mesh: Mesh, scope, fn):
    if any(mesh == m for m in scope):
        return
    for p in fn.params:
        if isinstance(p.type, TensorType) and isinstance(p.type.layout, ShardLayout):
            if p.type.layout.mesh == mesh:
                return
    raise VerifyError(
        f"{spec_ref_render(_PRIM_FUNCTION)}: ShardLayout references mesh {mesh!r} that is "
        f"not in current MeshScope stack nor bound by any param"
    )


def _verify_shape_of(expr: ShapeOf) -> None:
    if not isinstance(expr.param, Var):
        raise VerifyError(f"ShapeOf.param must be a Var, got {type(expr.param).__name__}")
    if not isinstance(expr.axis, int) or isinstance(expr.axis, bool) or expr.axis < 0:
        raise VerifyError(f"ShapeOf.axis must be a non-negative int, got {expr.axis!r}")
    expected = TensorType.scalar(dtype=DType.i32, storage=StorageKind.RMEM)
    if expr.type != expected:
        raise VerifyError(f"ShapeOf.type must be rank-0 i32 scalar TensorType, got {expr.type}")


def _verify_dispatch_call(stmt: DispatchCall, fn, module_fn_map, ctx):
    if len(stmt.subjects) != 1:
        raise VerifyError(f"DispatchCall: v0 requires len(subjects) == 1, got {len(stmt.subjects)}")
    subject = stmt.subjects[0]
    if not isinstance(subject, ShapeOf):
        raise VerifyError(
            f"DispatchCall: v0 dispatch subject must be ShapeOf(param, axis), "
            f"got {type(subject).__name__}"
        )
    _verify_shape_of(subject)

    if not any(subject.param is p for p in fn.params):
        raise VerifyError(
            f"DispatchCall: subject ShapeOf.param {subject.param.name!r} is not "
            f"one of the enclosing PrimFunction params"
        )
    pty = subject.param.type
    if not isinstance(pty, TensorType):
        raise VerifyError(
            f"DispatchCall: subject ShapeOf.param {subject.param.name!r} must "
            f"have TensorType, got {type(pty).__name__}"
        )
    if subject.axis >= len(pty.shape):
        raise VerifyError(
            f"DispatchCall: subject ShapeOf.axis {subject.axis} is out of "
            f"rank for param {subject.param.name!r} (rank={len(pty.shape)})"
        )
    if len(stmt.case_patterns) != len(stmt.case_calls):
        raise VerifyError(
            f"DispatchCall: len(case_patterns) {len(stmt.case_patterns)} != "
            f"len(case_calls) {len(stmt.case_calls)}"
        )
    for i, pats in enumerate(stmt.case_patterns):
        if len(pats) != len(stmt.subjects):
            raise VerifyError(
                f"DispatchCall: case_patterns[{i}] length {len(pats)} != "
                f"len(subjects) {len(stmt.subjects)}"
            )
        if not isinstance(pats[0], DimVarRangePat):
            raise VerifyError(
                f"DispatchCall: case_patterns[{i}][0] must be DimVarRangePat, "
                f"got {type(pats[0]).__name__}"
            )
    if not (
        isinstance(stmt.fallback, Sequential)
        and len(stmt.fallback.body) == 1
        and isinstance(stmt.fallback.body[0], Abort)
    ):
        raise VerifyError("DispatchCall: v0 fallback must be Sequential((Abort(),))")
    for call in stmt.case_calls:
        if not (isinstance(call, Evaluate) and isinstance(call.callable, SymbolRef)):
            inner = (
                f" (callable={type(call.callable).__name__})" if isinstance(call, Evaluate) else ""
            )
            raise VerifyError(
                "DispatchCall: each case call must be Evaluate(SymbolRef, args), "
                f"got {type(call).__name__}{inner}"
            )
        _verify_symbol_call(call, fn, module_fn_map, ctx)


def _resolve_symbol_ref(name, module_fn_map):
    """Resolve a SymbolRef name to its unique module ``PrimFunction``.

    Zero or more than one same-name match is an error — the same
    uniqueness contract as ``Module.lookup``.
    """
    matches = module_fn_map.get(name, ())
    if len(matches) != 1:
        raise VerifyError(
            f"SymbolRef {name!r} must resolve to exactly one module function, found {len(matches)}"
        )
    return matches[0]


def _verify_symbol_call(stmt: Evaluate, fn, module_fn_map, ctx):
    ref = stmt.callable
    if ref.nested:
        raise VerifyError(
            f"SymbolRef {ref.name!r}: nested {ref.nested!r} must be empty "
            f"(the module holds only top-level functions)"
        )
    if not module_fn_map:
        raise VerifyError(
            f"Evaluate(SymbolRef {ref.name!r}) requires module context; "
            f"call verify_prim_function(fn, module_fns=...) or verify_module([...])"
        )
    callee = _resolve_symbol_ref(ref.name, module_fn_map)
    expected_type = callable_type_for_prim_function(callee)
    if ref.type != expected_type:
        raise VerifyError(
            f"SymbolRef {ref.name!r}: type {ref.type} != resolved callee "
            f"CallableType {expected_type}"
        )
    if callee is fn:
        raise VerifyError("Evaluate(SymbolRef): recursion not allowed")
    _assert_no_path_back(callee, fn, module_fn_map, visited=set())
    if len(stmt.args) != len(callee.params):
        raise VerifyError(
            f"Evaluate(SymbolRef {ref.name!r}): arg count {len(stmt.args)} != "
            f"param count {len(callee.params)}"
        )
    for i, (arg, param) in enumerate(zip(stmt.args, callee.params)):
        arg_ty = ctx.type_of(arg)
        param_ty = param.type
        if type(arg_ty) is not type(param_ty):
            raise VerifyError(
                f"Evaluate(SymbolRef {ref.name!r}): arg[{i}] type kind "
                f"{type(arg_ty).__name__} != param type kind {type(param_ty).__name__}"
            )
        if arg_ty != param_ty:
            raise VerifyError(
                f"Evaluate(SymbolRef {ref.name!r}): arg[{i}] type {arg_ty} != "
                f"param {param.name!r} type {param_ty}"
            )


_LAUNCH_EXTENT_OPS = (DimAdd, DimSub, DimMul, DimFloorDiv, DimMod, DimMin, DimMax)


class _LaunchExtentVisitor(ExprVisitor[None]):
    def __init__(self, extent_params, ref_name) -> None:
        super().__init__()
        self.extent_params = extent_params
        self.ref_name = ref_name

    def visit_Constant(self, extent: Constant, ctx=None) -> None:
        if isinstance(extent.value, bool) or not isinstance(extent.value, int):
            raise VerifyError(
                f"Launch of {self.ref_name!r}: grid/block extent Constant must be an "
                f"int, got {extent.value!r}"
            )

    def visit_ShapeOf(self, extent: ShapeOf, ctx=None) -> None:
        param = self.extent_params.get(id(extent.param))
        if param is None:
            raise VerifyError(
                f"Launch of {self.ref_name!r}: grid/block extent ShapeOf references "
                f"{extent.param.name!r}, which is not a forwarded launch "
                f"argument or host parameter"
            )
        rank = len(param.type.shape) if isinstance(param.type, TensorType) else 0
        if not isinstance(param.type, TensorType) or not (0 <= extent.axis < rank):
            raise VerifyError(
                f"Launch of {self.ref_name!r}: grid/block extent ShapeOf axis "
                f"{extent.axis} is out of range for {extent.param.name!r} "
                f"(rank {rank})"
            )

    def visit_Call(self, extent: Call, ctx=None) -> None:
        if not (isinstance(extent.target, _LAUNCH_EXTENT_OPS)):
            self.default_visit(extent, ctx)
            return
        self.visit_operands(extent, ctx)

    def default_visit(self, extent, ctx=None) -> None:
        raise VerifyError(
            f"Launch of {self.ref_name!r}: grid/block extent must be an integer "
            f"Constant, ShapeOf, or dim-arithmetic Call, got {type(extent).__name__}"
        )


def _verify_launch_extent(extent, extent_params, ref_name) -> None:
    """Verify launch extent.

    A grid/block launch extent Expr is an integer ``Constant``, a
    ``ShapeOf`` of a forwarded / entry tensor parameter, or a dim-arithmetic
    ``Call`` whose operands are themselves valid extents.
    """
    if isinstance(extent, int) and not isinstance(extent, bool):
        return
    _LaunchExtentVisitor(extent_params, ref_name).visit(extent)


def _verify_launch(stmt: Evaluate, fn, module_fn_map, ctx):
    """Host-launch checks; full placement validation lands in CUDA lowering.

    Host-launch checks; full placement validation lands in CUDA lowering
    (a launch tensor arg's storage vs the runtime device type).

    The launch is ``Evaluate(Launch(...), (SymbolRef(callee), grid_x, grid_y,
    grid_z, block_x, block_y, block_z, *forwarded))``.
    """
    ref = stmt.args[0]
    if not isinstance(ref, SymbolRef):
        raise VerifyError(f"Launch: args[0] must be a SymbolRef callee, got {type(ref).__name__}")
    if ref.nested:
        raise VerifyError(
            f"Launch callee SymbolRef {ref.name!r}: nested {ref.nested!r} must "
            f"be empty (the module holds only top-level functions)"
        )
    if len(stmt.args) < 7:
        raise VerifyError(
            f"Launch of {ref.name!r}: expected at least 7 args (callee + six "
            f"grid/block extents), got {len(stmt.args)}"
        )
    forwarded = stmt.args[7:]

    extent_params = {id(p): p for p in fn.params}
    for a in forwarded:
        if isinstance(a, Var):
            extent_params.setdefault(id(a), a)
    for extent in stmt.args[1:7]:
        _verify_launch_extent(extent, extent_params, ref.name)
    for i, arg in enumerate(forwarded):
        if not isinstance(ctx.type_of(arg), TensorType):
            raise VerifyError(f"Launch of {ref.name!r}: forwarded arg[{i}] must be a tensor")
    if not module_fn_map:
        return
    callee = _resolve_symbol_ref(ref.name, module_fn_map)
    if not isinstance(callee.target, CudaTarget):
        raise VerifyError(f"Launch callee {callee.name!r} must target a CUDA device")
    expected = callable_type_for_prim_function(callee)
    if ref.type != expected:
        raise VerifyError(
            f"Launch callee SymbolRef {ref.name!r}: type {ref.type} != resolved "
            f"callee CallableType {expected}"
        )

    for p in callee.params:
        if not is_shape_scalar(p):
            continue
        parsed = parse_shape_var_name(p.name)
        if parsed is None:
            continue
        base, axis = parsed
        bt = next(
            (
                q
                for q in callee.params
                if q.name == base and isinstance(q.type, TensorType) and q.type.shape
            ),
            None,
        )
        if bt is not None and not (0 <= axis < len(bt.type.shape)):
            raise VerifyError(
                f"Launch of {callee.name!r}: shape scalar {p.name!r} references "
                f"axis {axis} of {base!r}, which has rank {len(bt.type.shape)}"
            )

    visible = [p for p in callee.params if not is_hidden_shape_scalar(p, callee.params)]
    if len(forwarded) != len(visible):
        raise VerifyError(
            f"Launch of {callee.name!r}: forwarded arg count {len(forwarded)} != "
            f"visible param count {len(visible)} (hidden shape scalars are "
            f"derived host-side, not passed)"
        )

    visible_tensors = {p.name for p in visible if isinstance(p.type, TensorType) and p.type.shape}
    for p in callee.params:
        if not is_hidden_shape_scalar(p, callee.params):
            continue
        base, _axis = parse_shape_var_name(p.name)
        if base not in visible_tensors:
            raise VerifyError(
                f"Launch of {callee.name!r}: hidden shape scalar {p.name!r} "
                f"derives from {base!r}, which is not a launched tensor argument"
            )


def _assert_no_path_back(callee, root, module_fn_map, visited):
    if callee is root:
        raise VerifyError(f"call DAG violation: {callee.name!r} calls back into {root.name!r}")
    if callee.name in visited:
        return
    visited.add(callee.name)
    for s in _iter_all_stmts(callee.body):
        if isinstance(s, Evaluate) and isinstance(s.callable, SymbolRef):
            matches = module_fn_map.get(s.callable.name, ())
            if len(matches) == 1:
                _assert_no_path_back(matches[0], root, module_fn_map, visited)


def _iter_all_stmts(body):
    """Yield every Stmt reachable from a Sequential / any Stmt tree."""
    stack = [body]
    while stack:
        s = stack.pop()
        yield s
        if isinstance(s, Sequential):
            stack.extend(s.body)
        elif isinstance(s, (For, While, MeshScope, LetStmt)):
            stack.append(s.body)
        elif isinstance(s, If):
            stack.append(s.then_body)
            stack.append(s.else_body)
        elif isinstance(s, DispatchCall):
            stack.extend(s.case_calls)
            stack.append(s.fallback)


def verify_module(fns) -> None:
    """V1 helper: fns is a list mixing hir.Function and tir.PrimFunction.

    Each name maps to exactly one entry. Specialization variants live on a
    dispatch prototype's ``variants`` field
    ([hir §1.1](docs/spec/hir.md#11-function)), never as separate
    same-name entries. Sealed-module structural invariants: a top-level entry
    is never a variant, and every entry is callable (``body is None`` only when
    it carries variants).
    """
    prim_fns = [f for f in fns if isinstance(f, PrimFunction)]
    for f in fns:
        if isinstance(f, HirFunction):
            verify_function(f)
            if f.specializations:
                raise VerifyError(
                    f"verify_module: {f.name!r} is a specialization variant at "
                    f"top level; variants live on the base function's "
                    f"`variants`, not as Module.functions entries"
                )
            if f.body is None and not f.variants:
                raise VerifyError(
                    f"verify_module: {f.name!r} has no body and no variants; a "
                    f"module function must be callable"
                )
        elif isinstance(f, PrimFunction):
            verify_prim_function(f, module_fns=prim_fns)
        else:
            raise VerifyError(f"verify_module: unknown function type {type(f).__name__}")

    by_name: dict[str, list] = {}
    for f in fns:
        by_name.setdefault(f.name, []).append(f)
    for name, group in by_name.items():
        if len(group) > 1:
            raise VerifyError(
                f"verify_module: {name!r} appears {len(group)} times; one name "
                f"maps to one function (specialization variants live on "
                f"Function.variants)"
            )


__all__ = ["verify_prim_function", "verify_module"]
