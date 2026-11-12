"""Verify HIR and TIR functions, bindings, calls, and contextual placement.

The checked rule set is defined by
[tir §1.3](docs/spec/tir.md#13-primfunction).
"""

from __future__ import annotations

import dataclasses
from contextlib import ExitStack, contextmanager

from tilefoundry import services, utils
from tilefoundry.interfaces import typeinfer, valuation, verify
from tilefoundry.ir import core, dim, expr, pattern, tir, types
from tilefoundry.ir.tir import memory
from tilefoundry.target import cpu

PRIM_FUNCTION = "[tir §1.3](docs/spec/tir.md#13-primfunction)"


verify_value_layouts = verify.require_layouts
"""Require concrete tensor layouts on values, independently of annotations."""


def verify_function(fn: core.Function, *, module=None) -> None:
    """Verify HIR parameters, symbolic dimensions, body, and variants."""
    for p in fn.params:
        if not isinstance(p, expr.Var):
            raise expr.VerifyError(f"hir Function {fn.name!r}: params must be Vars")
    verify_signature_dim_vars(fn)
    if fn.variants:
        if fn.body is not None:
            raise expr.VerifyError(
                f"hir Function {fn.name!r}: a function with variants must have "
                f"no body (a dispatch prototype's body is None / `pass`)"
            )
        verify_variants(fn, module=module)
        return
    if fn.body is None:
        return
    if not isinstance(fn.body, expr.Expr):
        raise expr.VerifyError(
            f"hir Function {fn.name!r}: body must be an Expr or None, got {type(fn.body).__name__}"
        )
    reject_stmt_nodes(fn.body)
    for value in (*fn.params, fn.body):
        verify_value_layouts(value)


def verify_variants(base: core.Function, *, module=None) -> None:
    """Verify a dispatch prototype's variants and their envelope partition."""
    base_param_types = tuple(p.annotation for p in base.params)
    sigs: dict[str, core.Function] = {}
    for v in base.variants:
        if v.variants:
            raise expr.VerifyError(
                f"hir Function {v.name!r}: a variant must not carry variants "
                f"(specialization nesting is one level)"
            )
        if not v.specializations:
            raise expr.VerifyError(
                f"hir Function {base.name!r}: a variant must carry a specialization pattern"
            )
        if v.body is None:
            raise expr.VerifyError(
                f"hir Function {base.name!r}: a variant must have a real body, not `pass`"
            )
        if v.name != base.name:
            raise expr.VerifyError(f"variant name {v.name!r} does not match base {base.name!r}")
        if tuple(p.annotation for p in v.params) != base_param_types or (
            v.return_annotation != base.return_annotation
        ):
            raise expr.VerifyError(
                f"hir Function {base.name!r}: variant signature must match the base signature"
            )
        sig = valuation.canonical_specialization_signature(v.specializations)
        if sig in sigs:
            raise expr.VerifyError(
                f"hir Function {base.name!r}: duplicate variant canonical signature {sig!r}"
            )
        sigs[sig] = v
        verify_function(v, module=module)
    verify_partition(base)


def verify_partition(base: core.Function) -> None:
    """Verify partition.

    Check the variants' ranges partition the base DimVar envelope —
    pairwise disjoint and jointly complete over closed specialization ranges.
    """
    dim_vars: set[str] = set()
    ranges: list[tuple[int, int]] = []
    for v in base.variants:
        for pat in v.specializations:
            if not isinstance(pat, pattern.Range):
                raise expr.VerifyError(
                    f"hir Function {base.name!r}: only pattern.Range is "
                    f"supported for dispatch (got {type(pat).__name__})"
                )
            dim_vars.add(pat.dim_var)
            ranges.append((pat.lo, pat.hi))
    if len(dim_vars) != 1:
        raise expr.VerifyError(
            f"hir Function {base.name!r}: variants must dispatch on a single "
            f"DimVar, got {sorted(dim_vars)}"
        )
    envelope = collect_param_dim_vars(base).get(next(iter(dim_vars)))
    if envelope is None:
        raise expr.VerifyError(
            f"hir Function {base.name!r}: dispatch DimVar "
            f"{next(iter(dim_vars))!r} is not reachable from an input parameter"
        )
    lo, hi = envelope

    cursor = lo
    for rlo, rhi in sorted(ranges):
        if rlo != cursor:
            raise expr.VerifyError(
                f"hir Function {base.name!r}: variant ranges do not partition "
                f"envelope [{lo}, {hi}] — gap or overlap at {rlo} (expected "
                f"{cursor})"
            )
        cursor = rhi + 1
    if cursor != hi + 1:
        raise expr.VerifyError(
            f"hir Function {base.name!r}: variant ranges cover "
            f"[{lo}, {cursor - 1}] but the envelope is [{lo}, {hi}]"
        )


def ingest_dim_vars(ty: object, bounds: dict[str, tuple[int, int]]) -> None:
    """Collect consistent ``DimVar`` bounds recursively from a signature type.

    Tensor tuples and dimension-expression calls are traversed. Reusing a name
    with different bounds raises ``VerifyError``.

    See [types §4](docs/spec/types.md#4-dim--symbolic-shape-dimensions).
    """
    if isinstance(ty, types.TensorType):
        for entry in ty.shape:
            ingest_shape_entry(entry, bounds)
    elif isinstance(ty, types.TupleType):
        for field in ty.fields:
            ingest_dim_vars(field, bounds)


class SignatureDimVisitor(core.ExprVisitor[None]):
    def __init__(self, bounds: dict[str, tuple[int, int]]) -> None:
        super().__init__()
        self.bounds = bounds

    def visit_DimVar(self, entry: expr.DimVar, ctx=None) -> None:
        prior = self.bounds.get(entry.name)
        if prior is None:
            self.bounds[entry.name] = (entry.lo, entry.hi)
        elif prior != (entry.lo, entry.hi):
            raise expr.VerifyError(
                f"inconsistent DimVar bounds for {entry.name!r} within "
                f"function signature: [{prior[0]}, {prior[1]}] vs "
                f"[{entry.lo}, {entry.hi}]"
            )

    def visit_Call(self, entry: expr.Call, ctx=None) -> None:
        for arg in entry.args:
            self.visit(arg, ctx)

    def default_visit(self, entry, ctx=None) -> None:
        return None


def ingest_shape_entry(entry: object, bounds: dict[str, tuple[int, int]]) -> None:
    if isinstance(entry, (expr.DimVar, expr.Call)):
        SignatureDimVisitor(bounds).visit(entry)


def collect_param_dim_vars(fn: core.Function) -> dict[str, tuple[int, int]]:
    """Bounds from ``fn.params`` only.

    Drives envelope ⊆ and unknown-name checks: specializations must
    anchor to a ``DimVar`` reachable from an *input* param, because
    Dispatch subjects lower to ``ShapeOf(param, axis)`` and
    can only reference a value the caller provides.
    """
    bounds: dict[str, tuple[int, int]] = {}
    for p in fn.params:
        ingest_dim_vars(p.type, bounds)
    return bounds


def check_signature_dim_var_consistency(fn: core.Function) -> None:
    """Scan params + return_type for same-name ``DimVar`` consistency.

    Recurses through ``TensorType`` and ``TupleType``. Raises on any
    same-name disagreement anywhere in the signature.
    """
    bounds: dict[str, tuple[int, int]] = {}
    for p in fn.params:
        ingest_dim_vars(p.type, bounds)
    ingest_dim_vars(getattr(fn, "return_type", None), bounds)


def verify_signature_dim_vars(fn: core.Function) -> None:

    check_signature_dim_var_consistency(fn)
    param_bounds = collect_param_dim_vars(fn)
    for pat in fn.specializations:
        if not isinstance(pat, pattern.Range):
            continue
        dv_bounds = param_bounds.get(pat.dim_var)
        if dv_bounds is None:
            raise expr.VerifyError(
                f"specialization pattern.Range({pat.dim_var!r}, {pat.lo}, "
                f"{pat.hi}) references unknown DimVar (specializations must "
                f"anchor to a DimVar reachable from an input parameter)"
            )
        lo, hi = dv_bounds
        if not (lo <= pat.lo and pat.hi <= hi):
            raise expr.VerifyError(
                f"pattern.Range ({pat.dim_var!r}, {pat.lo}, {pat.hi}) is not "
                f"contained in DimVar envelope [{lo}, {hi}]"
            )


class StmtRejectingVisitor(core.ExprVisitor[None]):
    def walk_region(self, node: core.LoopRegion | core.MeshRegion, ctx=None) -> None:
        valuation.verify_region_isolated(node)
        for operand in core.expr_children(node):
            self.visit(operand, ctx)

    visit_LoopRegion = walk_region
    visit_MeshRegion = walk_region

    def visit_Call(self, node: expr.Call, ctx=None) -> None:
        for arg in node.args:
            if isinstance(arg, core.Stmt):
                raise expr.VerifyError(
                    f"hir body contains a Stmt node {type(arg).__name__}; hir is expr-only"
                )
            self.visit(arg, ctx)

    def default_visit(self, node, ctx=None) -> None:
        if isinstance(node, core.Stmt):
            raise expr.VerifyError(
                f"hir body contains a Stmt node {type(node).__name__}; hir is expr-only"
            )


def reject_stmt_nodes(node: expr.Expr) -> None:
    StmtRejectingVisitor().visit(node)


def input_params(op_type: type) -> tuple:
    return tuple(param for param in expr.schema_of(op_type).signature if param.kind == "input")


def verify_prim_function(fn: core.PrimFunction, *, frame: valuation.Context | None = None) -> None:
    """Per [tir §1.3](docs/spec/tir.md#13-primfunction)'s rule list.

    *frame* is the Module, target and topologies that declare *fn*; a variant
    is checked in the frame of its prototype. With no frame, a ``Launch``
    callee is not resolved, and an ``Evaluate(FunctionRef)`` is rejected
    because it requires module context.
    """
    for param in fn.params:
        verify_value_layouts(param)
    check_param_homogeneity(fn)
    for variant in fn.variants:
        if len(variant.specializations) != 1 or not isinstance(
            variant.specializations[0], pattern.Range
        ):
            raise expr.VerifyError(f"PrimFunction {fn.name!r}: each variant must have one pattern.Range")
        pat = variant.specializations[0]
        if pattern.locate_dim_var(fn.params, pat.dim_var) is None:
            raise expr.VerifyError(
                f"PrimFunction {fn.name!r}: specialization subject {pat.dim_var!r} "
                "cannot be derived from parameters"
            )
        verify_prim_function(variant, frame=frame)

    ctx = verify.Context()
    if frame is not None:
        ctx = verify.Context(
            module=frame.module, function=fn, target=frame.target, topologies=frame.topologies
        )
    walk_stmt(fn.body, ctx, [], fn, Binders(fn.params))


def check_param_homogeneity(fn: core.PrimFunction) -> None:
    """Param homogeneity ([tir §1.3](docs/spec/tir.md#13-primfunction)).

    Param homogeneity ([tir §1.3](docs/spec/tir.md#13-primfunction)): all params' layouts must be
    consistently ShardLayout or all non-ShardLayout. Mixed is error.
    """
    kinds = []
    for p in fn.params:
        ty = p.type
        if not isinstance(ty, types.TensorType):
            continue
        kinds.append(isinstance(ty.layout, types.ShardLayout))
    if kinds and len(set(kinds)) > 1:
        raise expr.VerifyError(
            f"PrimFunction {fn.name!r}: {utils.spec_ref_render(PRIM_FUNCTION)} param layouts "
            f"must be homogeneously ShardLayout or non-ShardLayout, got mixed"
        )


class Binders:
    def __init__(self, params):
        self.visible = {id(param) for param in params}
        self.ever_bound = set(self.visible)

    def uses(self, value, stmt):
        if isinstance(value, expr.Var):
            if id(value) not in self.visible:
                raise expr.VerifyError(f"{value.name} is used outside the scope that binds it")
        elif isinstance(value, (expr.Op, core.PrimFunction, core.FunctionRef, core.ModuleList)):
            return
        elif isinstance(value, expr.Call):
            self.uses((value.target, value.args), stmt)
        elif isinstance(value, (expr.Expr, tuple, list)):
            children = value if isinstance(value, (tuple, list)) else core.expr_children(value)
            for child in children:
                self.uses(child, stmt)
        else:
            raise expr.VerifyError(f"non-IR node {type(value).__name__} in {type(stmt).__name__}")

    @contextmanager
    def bind(self, var, stmt):
        if id(var) in self.ever_bound:
            raise expr.VerifyError(
                f"{type(stmt).__name__} binding {var.name!r}: "
                f"{utils.spec_ref_render(PRIM_FUNCTION)} var must be a fresh Var; this Var "
                "is already bound in an outer scope"
            )
        self.ever_bound.add(id(var))
        self.visible.add(id(var))
        try:
            yield
        finally:
            self.visible.remove(id(var))


def walk_stmt(stmt, ctx, scope, fn, binders: Binders):
    for _, value in iter_stmt_expr_fields(stmt):
        if isinstance(value, expr.Expr):
            verify_value_layouts(value)
    match stmt:
        case core.Sequential():
            with ExitStack() as bound:
                for s in stmt.body:
                    walk_stmt(s, ctx, scope, fn, binders)
                    if isinstance(s, core.LetStmt):
                        bound.enter_context(binders.bind(s.var, s))
            return
        case core.LetStmt():
            binders.uses(stmt.value, stmt)
            value_ty = ctx.type_of(stmt.value)
            if stmt.var.type != value_ty:
                raise expr.VerifyError(
                    f"LetStmt binding {stmt.var.name!r}: var.type {stmt.var.type} "
                    f"!= value.type {value_ty}"
                )

            reject_nested_alloc_tensor(stmt.value, at_letstmt_value=True)
            check_embedded_sharding(stmt.value, scope, fn)
            return
        case core.For():
            binders.uses((stmt.start, stmt.stop, stmt.step), stmt)
            check_rank0_int(ctx, stmt, stmt.start, "For.start")
            check_rank0_int(ctx, stmt, stmt.stop, "For.stop")
            check_rank0_int(ctx, stmt, stmt.step, "For.step")
            for field, bound in (
                ("start", stmt.start),
                ("stop", stmt.stop),
                ("step", stmt.step),
            ):
                check_bound_coordinates(field, bound, scope)
            if isinstance(stmt.step, expr.Constant) and stmt.step.value == 0:
                raise expr.VerifyError("For.step must not be 0")
            iv_ty = stmt.induction_var.type
            if not (
                isinstance(iv_ty, types.TensorType)
                and iv_ty.shape == ()
                and iv_ty.dtype in (expr.DType.i32, expr.DType.i64)
            ):
                raise expr.VerifyError("For.induction_var must be rank-0 integer")
            with binders.bind(stmt.induction_var, stmt):
                walk_stmt(stmt.body, ctx, scope, fn, binders)
            return
        case core.While():
            binders.uses(stmt.cond, stmt)
            check_rank0_bool(ctx, stmt, stmt.cond)
            walk_stmt(stmt.body, ctx, scope, fn, binders)
            return
        case core.If():
            binders.uses(stmt.cond, stmt)
            check_rank0_bool(ctx, stmt, stmt.cond)
            walk_stmt(stmt.then_body, ctx, scope, fn, binders)
            walk_stmt(stmt.else_body, ctx, scope, fn, binders)
            return
        case core.MeshScope():
            scope.append(stmt.mesh)
            with binders.bind(stmt.binding, stmt):
                walk_stmt(stmt.body, ctx, scope, fn, binders)
            scope.pop()
            return
        case core.Return():
            for _, value in iter_stmt_expr_fields(stmt):
                binders.uses(value, stmt)
            return
        case core.Evaluate():
            binders.uses((stmt.callable, stmt.args), stmt)
            if isinstance(stmt.callable, tir.launch.Launch):
                verify_launch(stmt, fn, ctx)
            elif isinstance(stmt.callable, core.FunctionRef):
                verify_function_call(stmt, fn, ctx)
            elif isinstance(stmt.callable, (core.PrimFunction, core.ModuleList, expr.Call)):
                verify_retained_call(stmt, fn, ctx)
            else:
                op = stmt.callable
                ctx.enter_scope(scope)
                services.verify_call(op, stmt.args, ctx.current_mesh, frame=ctx)

            for arg in stmt.args:
                reject_nested_alloc_tensor(arg, at_letstmt_value=False)
                check_embedded_sharding(arg, scope, fn)
            return

    fn_verify = verify.registry.get(type(stmt), None)
    if fn_verify is None:
        raise expr.VerifyError(f"no verify_stmt registered for {type(stmt).__name__}")
    fn_verify(stmt, ctx)
    for field_name, field_val in iter_stmt_expr_fields(stmt):
        binders.uses(field_val, stmt)
        reject_nested_alloc_tensor(field_val, at_letstmt_value=False)
        check_embedded_sharding(field_val, scope, fn)


def iter_stmt_expr_fields(stmt):
    if not dataclasses.is_dataclass(stmt):
        return
    for f in dataclasses.fields(stmt):
        val = getattr(stmt, f.name)
        if f.type in (expr.Expr, "Expr") or isinstance(val, expr.Expr):
            yield f.name, val


def check_rank0_int(ctx, stmt, node, field: str):
    t = ctx.type_of(node)
    if not (isinstance(t, types.TensorType) and t.shape == () and t.dtype in (expr.DType.i32, expr.DType.i64)):
        raise expr.VerifyError(f"{field} must be rank-0 integer, got {t}")


def check_rank0_bool(ctx, stmt, node):
    t = ctx.type_of(node)
    if not (isinstance(t, types.TensorType) and t.shape == () and t.dtype == expr.DType.bool):
        raise expr.VerifyError(f"condition must be rank-0 bool, got {t}")


class AllocTensorRejectingVisitor(core.ExprVisitor[None]):
    def __init__(self, *, at_letstmt_value: bool) -> None:
        super().__init__()
        self.at_letstmt_value = at_letstmt_value

    def visit_Call(self, node: expr.Call, ctx=None) -> None:
        if isinstance(node.target, memory.AllocTensor):
            if not self.at_letstmt_value:
                raise expr.VerifyError(
                    f"{utils.spec_ref_render(PRIM_FUNCTION)}: Call(AllocTensor, ...) may only appear "
                    f"as a direct LetStmt.value; found nested inside another Expr"
                )
            self.at_letstmt_value = False
            self.visit_operands(node, ctx)
            return
        self.at_letstmt_value = False
        self.visit_operands(node, ctx)

    def default_visit(self, node, ctx=None) -> None:
        return None


def reject_nested_alloc_tensor(node: expr.Expr, *, at_letstmt_value: bool) -> None:
    """Reject an ``AllocTensor`` call nested outside a ``LetStmt`` value.

    ``Call(AllocTensor, ...)`` may appear only as ``LetStmt.value``.

    See [tir §1.3](docs/spec/tir.md#13-primfunction).
    """
    visitor = AllocTensorRejectingVisitor(at_letstmt_value=at_letstmt_value)
    visitor.visit(node)


def check_bound_coordinates(field: str, bound, scope) -> None:
    """Hold a loop bound's mesh coordinates to the scopes around the loop.

    A coordinate is a coordinate of one scope, read through that scope's
    binding, so a bound naming one has to stand inside a ``MeshScope`` of the
    mesh it names -- the enclosing scope is what says which unit it is about.
    """
    for node in core.collect_exprs(bound):
        if not (isinstance(node, expr.Call) and isinstance(node.target, dim.MeshCoord)):
            continue
        mesh = node.target.mesh
        axes = dim.literal_axes(node.args[0]) if node.args else None
        rank = len(types.flatten(mesh.layout).shape)
        if axes is None or not all(0 <= axis < rank for axis in axes):
            raise expr.VerifyError(f"For.{field} reads axes {axes!r} of a mesh of rank {rank}")
        if not any(held is mesh or held == mesh for held in scope):
            raise expr.VerifyError(
                f"For.{field} reads a coordinate of {mesh!r}, which no enclosing MeshScope binds"
            )


def check_embedded_sharding(node: expr.Expr, scope, fn):
    """Require embedded shard layouts to use a scoped or parameter mesh.

    See [tir §1.3](docs/spec/tir.md#13-primfunction).
    """
    to_visit = [node]
    while to_visit:
        e = to_visit.pop()
        if isinstance(e, expr.Call):
            op = e.target
            for attr_name, attr_val in iter_op_attrs(op):
                if isinstance(attr_val, types.ShardLayout):
                    assert_mesh_in_scope(attr_val.mesh, scope, fn)
            to_visit.extend(e.args)


def iter_op_attrs(op):
    for info in expr.parameters_of(type(op)):
        if info.kind == "attribute":
            yield info.name, getattr(op, info.name, None)


def assert_mesh_in_scope(mesh: types.Mesh, scope, fn):
    if any(mesh == m for m in scope) or (scope and mesh == types.Mesh.from_meshes(*scope)):
        return
    for p in fn.params:
        if isinstance(p.type, types.TensorType) and isinstance(p.type.layout, types.ShardLayout):
            if p.type.layout.mesh == mesh:
                return
    raise expr.VerifyError(
        f"{utils.spec_ref_render(PRIM_FUNCTION)}: ShardLayout references mesh {mesh!r} that is "
        f"not in current MeshScope stack nor bound by any param"
    )


def verify_shape_of(node: core.ShapeOf) -> None:
    if not isinstance(node.param, expr.Var):
        raise expr.VerifyError(f"ShapeOf.param must be a Var, got {type(node.param).__name__}")
    if not isinstance(node.axis, int) or isinstance(node.axis, bool) or node.axis < 0:
        raise expr.VerifyError(f"ShapeOf.axis must be a non-negative int, got {node.axis!r}")
    expected = types.TensorType.from_scalar(dtype=expr.DType.i32, storage=expr.StorageKind.RMEM)
    if node.type != expected:
        raise expr.VerifyError(f"ShapeOf.type must be rank-0 i32 scalar TensorType, got {node.type}")


def callee_frame(ref, frame):
    """The function *ref* names from *frame*, and the frame of the Module that declares it."""
    scope = None if frame is None else frame.module
    try:
        module, callee = core.resolve(scope, ref)
    except ValueError as error:
        raise expr.VerifyError(str(error)) from error
    target, topologies = frame.target, frame.topologies
    for node in core.descend(scope, ref):
        target = node.target if node.target is not None else target
        topologies = node.topologies if node.topologies is not None else topologies
    return callee, dataclasses.replace(
        frame, module=module, function=callee, target=target, topologies=topologies
    )


def verify_function_call(stmt: core.Evaluate, fn, ctx):
    ref = stmt.callable
    if ctx.module is None:
        raise expr.VerifyError(
            f"Evaluate(FunctionRef {ref.name!r}) requires module context; "
            f"call verify_prim_function(fn, frame=...) or verify_module(...)"
        )
    callee, frame = callee_frame(ref, ctx)
    if callee is fn:
        raise expr.VerifyError("Evaluate(FunctionRef): recursion not allowed")
    assert_no_path_back(callee, fn, visited=set(), frame=frame)
    verify_call_arguments(stmt, callee.params, f"Evaluate(FunctionRef {ref.name!r})", ctx)


def verify_call_arguments(stmt, params, label, ctx):
    if len(stmt.args) != len(params):
        raise expr.VerifyError(f"{label}: arg count {len(stmt.args)} != param count {len(params)}")
    for i, (arg, param) in enumerate(zip(stmt.args, params)):
        arg_ty = ctx.type_of(arg)
        param_ty = param.type
        if type(arg_ty) is not type(param_ty):
            raise expr.VerifyError(
                f"{label}: arg[{i}] type kind "
                f"{type(arg_ty).__name__} != param type kind {type(param_ty).__name__}"
            )
        if arg_ty != param_ty:
            raise expr.VerifyError(
                f"{label}: arg[{i}] type {arg_ty} != param {param.name!r} type {param_ty}"
            )


def retained_callee(target, frame):
    if not isinstance(target, (core.PrimFunction, core.ModuleList, expr.Call)):
        raise expr.VerifyError(f"unsupported TIR call target {type(target).__name__}")
    if not isinstance(target, core.PrimFunction):
        if frame is None:
            raise expr.VerifyError("instance TIR call requires module context")
        typeinfer.inference_type(target, typeinfer.TypeInferContext(module=frame.module, function=frame.function, target=frame.target, topologies=frame.topologies))
    inner = None if frame is None else frame.callee_context(expr.Call(target, (), type=types.UnitType()))
    callee = inner.function if inner is not None else target
    if not isinstance(callee, core.PrimFunction):
        raise expr.VerifyError(f"TIR instance entry {callee.name!r} is not a PrimFunction")
    return callee, inner


def verify_retained_call(stmt, fn, ctx):
    target = stmt.callable
    callee, frame = retained_callee(target, ctx)
    direct = isinstance(target, core.PrimFunction)
    label = (
        target.name
        if direct
        else (
            target.key
            if isinstance(target, core.ModuleList)
            else f"{target.args[0].key}[{services.render(target.args[1])}]"
        )
    )
    prefix = f"Evaluate({label})"
    assert_no_path_back(
        callee,
        fn,
        visited=set(),
        frame=frame,
        recursion=f"{prefix}: recursion not allowed",
    )
    params = tuple(param for param in callee.params if direct or not param.is_const)
    verify_call_arguments(stmt, params, prefix, ctx)


LAUNCH_EXTENT_OPS = (expr.DimAdd, expr.DimSub, expr.DimMul, expr.DimFloorDiv, expr.DimMod, expr.DimMin, expr.DimMax)


class LaunchExtentVisitor(core.ExprVisitor[None]):
    def __init__(self, extent_params, ref_name) -> None:
        super().__init__()
        self.extent_params = extent_params
        self.ref_name = ref_name

    def visit_Constant(self, extent: expr.Constant, ctx=None) -> None:
        if isinstance(extent.value, bool) or not isinstance(extent.value, int):
            raise expr.VerifyError(
                f"Launch of {self.ref_name!r}: grid/block extent Constant must be an "
                f"int, got {extent.value!r}"
            )

    def visit_ShapeOf(self, extent: core.ShapeOf, ctx=None) -> None:
        param = self.extent_params.get(id(extent.param))
        if param is None:
            raise expr.VerifyError(
                f"Launch of {self.ref_name!r}: grid/block extent ShapeOf references "
                f"{extent.param.name!r}, which is not a forwarded launch "
                f"argument or host parameter"
            )
        rank = len(param.type.shape) if isinstance(param.type, types.TensorType) else 0
        if not isinstance(param.type, types.TensorType) or not (0 <= extent.axis < rank):
            raise expr.VerifyError(
                f"Launch of {self.ref_name!r}: grid/block extent ShapeOf axis "
                f"{extent.axis} is out of range for {extent.param.name!r} "
                f"(rank {rank})"
            )

    def visit_Call(self, extent: expr.Call, ctx=None) -> None:
        if not (isinstance(extent.target, LAUNCH_EXTENT_OPS)):
            self.default_visit(extent, ctx)
            return
        self.visit_operands(extent, ctx)

    def default_visit(self, extent, ctx=None) -> None:
        raise expr.VerifyError(
            f"Launch of {self.ref_name!r}: grid/block extent must be an integer "
            f"Constant, ShapeOf, or dim-arithmetic Call, got {type(extent).__name__}"
        )


def verify_launch_extent(extent, extent_params, ref_name) -> None:
    """Verify launch extent.

    A grid/block launch extent Expr is an integer ``Constant``, a
    ``ShapeOf`` of a forwarded / entry tensor parameter, or a dim-arithmetic
    ``Call`` whose operands are themselves valid extents.
    """
    if isinstance(extent, int) and not isinstance(extent, bool):
        return
    LaunchExtentVisitor(extent_params, ref_name).visit(extent)


def verify_launch(stmt: core.Evaluate, fn, ctx):
    """Check host launches before target-specific placement validation.

    The launch is ``Evaluate(Launch(...), (FunctionRef(callee), grid_x, grid_y,
    grid_z, block_x, block_y, block_z, *forwarded))``. An HIR callee's arguments
    were written and checked by the host-entry pass that built the Launch
    ([passes §7.3](docs/spec/passes.md#73-inserthostentrypass)), so only
    its selector is resolved here.
    """
    ref = stmt.args[0]
    if not isinstance(ref, core.FunctionRef):
        raise expr.VerifyError(f"Launch: args[0] must be a FunctionRef callee, got {type(ref).__name__}")
    if len(stmt.args) < 7:
        raise expr.VerifyError(
            f"Launch of {ref.name!r}: expected at least 7 args (callee + six "
            f"grid/block extents), got {len(stmt.args)}"
        )
    forwarded = stmt.args[7:]

    extent_params = {id(p): p for p in fn.params}
    for a in forwarded:
        if isinstance(a, expr.Var):
            extent_params.setdefault(id(a), a)
    for extent in stmt.args[1:7]:
        verify_launch_extent(extent, extent_params, ref.name)
    for i, arg in enumerate(forwarded):
        if not isinstance(ctx.type_of(arg), types.TensorType):
            raise expr.VerifyError(f"Launch of {ref.name!r}: forwarded arg[{i}] must be a tensor")
    if ctx.module is None:
        return
    callee, _ = callee_frame(ref, ctx)
    if isinstance(callee, core.Function):
        return
    if isinstance(callee.target, cpu.Target):
        raise expr.VerifyError(f"Launch callee {callee.name!r} must target a device, not the CPU")

    if len(forwarded) != len(callee.params):
        raise expr.VerifyError(
            f"Launch of {callee.name!r}: forwarded arg count {len(forwarded)} != "
            f"param count {len(callee.params)}"
        )
    for i, (arg, param) in enumerate(zip(forwarded, callee.params)):
        arg_ty = ctx.type_of(arg)
        if arg_ty != param.type:
            raise expr.VerifyError(
                f"Launch of {callee.name!r}: forwarded arg[{i}] type {arg_ty} != "
                f"param {param.name!r} type {param.type}"
            )


def assert_no_path_back(callee, root, visited, frame=None, recursion=None):
    if callee is root:
        raise expr.VerifyError(
            recursion or f"call DAG violation: {callee.name!r} calls back into {root.name!r}"
        )
    key = (id(callee), id(frame.module) if frame is not None else None)
    if key in visited:
        return
    visited.add(key)
    for stmt in iter_all_stmts(callee.body):
        if not isinstance(stmt, core.Evaluate):
            continue
        target = stmt.callable
        if isinstance(target, core.FunctionRef):
            reached, inner = callee_frame(target, frame)
        elif isinstance(target, (core.PrimFunction, core.ModuleList, expr.Call)):
            reached, inner = retained_callee(target, frame)
        else:
            continue
        assert_no_path_back(reached, root, visited, inner, recursion)


def iter_all_stmts(body):
    """Yield every Stmt reachable from a Sequential / any Stmt tree."""
    stack = [body]
    while stack:
        s = stack.pop()
        yield s
        if isinstance(s, core.Sequential):
            stack.extend(s.body)
        elif isinstance(s, (core.For, core.While, core.MeshScope)):
            stack.append(s.body)
        elif isinstance(s, core.If):
            stack.append(s.then_body)
            stack.append(s.else_body)


def verify_module(fns) -> None:
    """Verify a Module tree or a legacy flat function iterable.

    Names are unique within each Module. Variants live on their prototype's
    `variants` field, not as separate same-name entries. A top-level entry is
    never an HIR variant and remains callable when it has no body.
    See [hir §1.1](docs/spec/hir.md#11-function).
    """
    root = fns
    if not isinstance(fns, core.Module):
        root = core.Module("<functions>", ())
        root.replace_functions(fns)
    frames = valuation.collect_functions(root)
    owners = {id(root): root, **{id(frame.module): frame.module for frame in frames.values()}}
    groups = tuple(node.functions for node in owners.values())
    for frame in frames.values():
        f = frame.function
        if isinstance(f, core.Function):
            verify_function(f)
            if f.specializations:
                raise expr.VerifyError(
                    f"verify_module: {f.name!r} is a specialization variant at "
                    f"top level; variants live on the base function's "
                    f"`variants`, not as Module.functions entries"
                )
            if f.body is None and not f.variants:
                raise expr.VerifyError(
                    f"verify_module: {f.name!r} has no body and no variants; a "
                    f"module function must be callable"
                )
        elif isinstance(f, core.PrimFunction):
            verify_prim_function(f, frame=frame)
        else:
            raise expr.VerifyError(f"verify_module: unknown function type {type(f).__name__}")

    for functions in groups:
        by_name: dict[str, list] = {}
        for f in functions:
            by_name.setdefault(f.name, []).append(f)
        for name, group in by_name.items():
            if len(group) > 1:
                raise expr.VerifyError(
                    f"verify_module: {name!r} appears {len(group)} times; one name "
                    f"maps to one local function (specialization variants live on "
                    f"Function.variants)"
                )


__all__ = [
    "input_params",
    "verify_function",
    "verify_module",
    "verify_prim_function",
    "verify_shape_of",
    "verify_value_layouts",
]
