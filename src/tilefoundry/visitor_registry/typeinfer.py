"""Type inference over expressions, functions, and authored regions."""

from __future__ import annotations

from dataclasses import MISSING, replace

from tilefoundry.ir.core.expr import Call, Constant, Expr, Tuple, Var
from tilefoundry.ir.core.metadata import (
    RangeMetadata,
    attach_metadata,
    detach_metadata,
)
from tilefoundry.ir.core.module import ModuleList
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.sharding.reshard import Reshard as HirReshard
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.mesh_scope import covered_by_scope, storage_reaches
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.types.callable_type import callable_type_for
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.dtype import IntegerDType
from tilefoundry.ir.types.mesh import make_mesh
from tilefoundry.ir.types.shard_layout import ShardLayout
from tilefoundry.ir.types.substitute import canonicalize_dims
from tilefoundry.ir.types.tensor_type import TensorType, TupleType, Type
from tilefoundry.ir.types.utils import participant_layout, types_compatible
from tilefoundry.ir.visitor import ExprVisitor, expr_children

from .contexts import CallFrame, TypeInferContext, TypeInferResults
from .registries import typeinfer_registry


def input_scope_refusal(operand_types, mesh) -> str | None:
    """The first input whose holder or storage does not reach its execution scope."""
    if mesh is None:
        return None
    for index, arg_type in enumerate(operand_types):
        layout = getattr(arg_type, "layout", None)
        if not isinstance(layout, ShardLayout):
            continue
        try:
            projected = participant_layout(layout)
            covered = covered_by_scope(projected.mesh, mesh)
        except ValueError as error:
            return str(error)
        if not covered:
            return (
                f"input {index} is laid out more finely than the scope it "
                "runs in; write it inside that scope, or reshard it back first"
            )
        if not storage_reaches(arg_type.storage, projected.mesh, mesh):
            return (
                f"input {index} is laid out more coarsely and kept in "
                f"{arg_type.storage.name.lower()}, which does not reach the units "
                "this runs on; reshard it to smem or gmem first"
            )
    return None


def _check_module_index(slot: ModuleList, index=MISSING, *, index_type=None):
    """Validate one entry call or subscript, returning its normalized index.

    Callers supply the current frame's slot when a tree has been rebuilt.
    Variable expressions remain expressions in the instance stack.
    """
    if index is MISSING:
        if slot.indexed:
            raise ValueError(f"ModuleList slot {slot.key!r} must be indexed before calling")
        return None
    if not slot.indexed:
        raise ValueError(f"single slot {slot.key!r} cannot be indexed")
    if isinstance(index, Expr) or index_type is not None:
        index_type = index.type if index_type is None else index_type

        if (
            not isinstance(index_type, TensorType)
            or index_type.shape != ()
            or not isinstance(index_type.dtype, IntegerDType)
        ):
            raise ValueError(f"ModuleList slot {slot.key!r}: index must be an integer scalar")
    if isinstance(index, Constant):
        index = index.value
    if isinstance(index, bool) or not isinstance(index, (int, Expr)):
        raise ValueError(f"ModuleList slot {slot.key!r}: index must be an integer scalar")
    if isinstance(index, int):
        if not 0 <= index < len(slot):
            raise ValueError(
                f"ModuleList slot {slot.key!r}: index {index!r} out of range for N={len(slot)}"
            )
    elif any(element is not slot.elements[0] for element in slot.elements[1:]):
        raise ValueError(f"ModuleList slot {slot.key!r}: variable index requires the same definition")
    return index



class TypeInferVisitor(ExprVisitor[Type]):
    """Derive one type for each ``Expr`` kind.

    See [hir §1.1](docs/spec/hir.md#11-function) and
    [visitor-registry §4](docs/spec/visitor-registry.md#4-instance-1--typeinfer).

    The context owns one identity memo for the current inference scope. A
    missing leaf raises through ``default_visit_leaf`` rather than trusting a
    stale ``expr.type``.
    """

    def __init__(self, *, memo=None, owns_body: bool = True, ranges: bool = False) -> None:
        super().__init__(memo=memo)
        self._memo_supplied = memo is not None
        self._visit_depth = 0
        self._owns_body = owns_body
        self._ranges = ranges

    def visit(self, expr: Expr, ctx: TypeInferContext) -> Type:
        """Derive one type while preserving the active execution domain."""
        outermost = self._visit_depth == 0
        if outermost:
            if self._memo_supplied:
                ctx = replace(ctx, memo=self._memo)
            else:
                self._memo = ctx.memo
        cached = id(expr) in self._memo
        self._visit_depth += 1
        try:
            results = super().visit(expr, ctx)
            if not isinstance(results, TypeInferResults):
                results = TypeInferResults(results)
            result = canonicalize_dims(results.type)
            self._memo[id(expr)] = (expr, result)
            if self._owns_body:
                expr.type = result
            if self._ranges and not cached:
                self._record_range(expr, results)
            return result
        finally:
            self._visit_depth -= 1

    def visit_operands(self, expr: Expr, ctx: TypeInferContext) -> tuple[Type, ...]:
        """Infer Expr operands while treating symbolic dimensions as meta-scalars."""
        return tuple(
            TensorType.umat_scalar() if isinstance(child, DimVar) else self.visit(child, ctx)
            for child in expr_children(expr)
        )

    @staticmethod
    def _record_range(expr: Expr, results: TypeInferResults) -> None:
        """Replace one expression's derived range with this inference result."""
        if results.value_range is None:
            detach_metadata(expr, RangeMetadata)
        else:
            attach_metadata(expr, RangeMetadata(*results.value_range))

    def visit_leaf_Var(self, var: Var, _operands, ctx: TypeInferContext) -> Type:
        return var.annotation

    def visit_leaf_Constant(self, c: Constant, _operands, ctx: TypeInferContext) -> Type:
        return c.type

    def visit_leaf_Call(self, call: Call, arg_types, ctx: TypeInferContext) -> Type:
        target = call.target
        if isinstance(target, TupleGetItem) and isinstance(call.args[0], ModuleList):
            slot = call.args[0]
            if ctx.frame is not None:
                slot = ctx.frame.module.children[slot.key]
            try:
                index = _check_module_index(slot, call.args[1], index_type=arg_types[1])
            except ValueError as error:
                ctx.error(call, str(error))
            return arg_types[0].fields[index if isinstance(index, int) else 0]
        if isinstance(target, Call):
            arg_types = arg_types[1:]
        child_frame = None
        slot = target if isinstance(target, ModuleList) else (
            target.args[0] if isinstance(target, Call) and isinstance(target.target, TupleGetItem)
            and isinstance(target.args[0], ModuleList) else None
        )
        if slot is not None:
            if ctx.frame is None:
                ctx.error(call, "instance call requires a module scope")
            try:
                slot = ctx.frame.module.children[slot.key]
                if isinstance(target, ModuleList):
                    _check_module_index(slot)
                    index = 0
                else:
                    index = target.args[1]
                    index = index.value if isinstance(index, Constant) else 0
                child = slot.elements[index]
                target = child.entry_function()
                child_frame = CallFrame(
                    child, target, child.target if child.target is not None else ctx.frame.target,
                    child.topologies if child.topologies is not None else ctx.frame.topologies,
                )
            except (KeyError, ValueError) as error:
                ctx.error(call, str(error))
        from tilefoundry.ir.hir.schedule import ScheduleOp  # noqa: PLC0415

        if not isinstance(target, (HirReshard, ScheduleOp)):
            refusal = input_scope_refusal(arg_types, ctx.current_mesh)
            if refusal is not None:
                ctx.error(call, refusal)
        if isinstance(target, Function):
            return self._call_function(call, target, arg_types, ctx, child_frame=child_frame)
        op_cls = type(target)
        fn = self._target_rule(op_cls, ctx) or typeinfer_registry.lookup(op_cls)
        if fn is None:
            ctx.error(call, f"no typeinfer registered for {op_cls.__name__}")
        return fn(call, ctx)

    @staticmethod
    def _target_rule(op_cls: type, ctx: TypeInferContext):
        """The rule registered for this Op under the walk's Target, if any."""
        target = ctx.resolve_target()
        if target is None:
            return None
        return typeinfer_registry.lookup((type(target), op_cls))

    def _call_function(
        self,
        call: Call,
        callee: Function,
        arg_types: tuple[Type, ...],
        ctx: TypeInferContext,
        *,
        child_frame: CallFrame | None = None,
    ) -> Type:
        child = child_frame is not None
        supplied = tuple(
            param for param in callee.params if not (child and param.is_const)
        )
        if len(arg_types) != len(supplied):
            kind = "activation(s)" if child else "parameter(s)"
            ctx.error(
                call,
                f"hir Function call {callee.name!r}: arity mismatch — "
                f"callee declares {len(supplied)} {kind}, call passed {len(arg_types)}",
            )

        given = iter(enumerate(arg_types))
        memo = {}
        for param in callee.params:
            if child and param.is_const:
                memo[id(param)] = (param, param.annotation)
                continue
            index, arg_type = next(given)
            declared = param.annotation
            if not types_compatible(declared, arg_type):
                ctx.error(
                    call,
                    f"hir Function call {callee.name!r}: arg {index} type mismatch — "
                    f"callee param {param.name!r} expects {declared!r}, got {arg_type!r}",
                )
            memo[id(param)] = (param, arg_type)

        if callee.body is None or callee.variants:
            return callee.return_type

        inner = replace(ctx, frame=child_frame, memo={}) if child else ctx.for_callee(callee)
        context = None if inner.frame is None else (
            id(inner.frame.module), inner.frame.target, inner.frame.topologies
        )
        key = (id(callee), arg_types, ctx.current_mesh, context)
        cached = ctx.instantiated_memo.get(key)
        if cached is not None:
            return cached
        result = TypeInferVisitor(memo=memo, owns_body=False, ranges=False).visit(
            callee.body, inner
        )
        ctx.instantiated_memo[key] = result
        return result

    def visit_leaf_ModuleList(self, slot, _operands, ctx: TypeInferContext) -> Type:
        try:
            current = ctx.frame.module.children[slot.key] if ctx.frame is not None else slot
            return current.type
        except (KeyError, ValueError) as error:
            ctx.error(slot, str(error))

    def visit_leaf_Tuple(self, tup: Tuple, operands, ctx: TypeInferContext) -> Type:
        """Visit Tuple.

        Structural: the field types of the (possibly just-elaborated)
        elements, never the node's own stamped ``.type`` ([hir §1.1](docs/spec/hir.md#11-function)).
        """
        return TupleType(fields=operands)

    def _region_memo(
        self, region: LoopRegion | MeshRegion, ctx: TypeInferContext
    ) -> dict[int, tuple[Expr, Type]]:
        """Infer arguments and bind each region parameter to its argument's type.

        A region parameter has no type of its own: it is the value its argument
        takes as the region is entered, so a type stored on it by an earlier
        walk does not constrain this one. A walk that owns the body stores the
        entry type on every parameter, captures the body never reads included.
        """
        arg_types = tuple(self.visit(arg, ctx) for arg in region.args)
        from .verify import verify_region_isolated  # noqa: PLC0415

        verify_region_isolated(region, ctx)
        if self._owns_body:
            for param, arg_type in zip(region.params, arg_types, strict=True):
                param.type = arg_type
        return {
            **ctx.memo,
            **{
                id(param): (param, arg_type)
                for param, arg_type in zip(region.params, arg_types, strict=True)
            },
        }

    def visit_LoopRegion(self, region: LoopRegion, ctx: TypeInferContext) -> Type:
        """Infer a loop after binding its induction and carried variables.

        The entry values give the carried types and the result, so a loop that
        runs no iteration has the type it was entered with. Each yielded value
        must fit the entry type of the parameter it carries into; a yield that
        is more specific does not change that type.
        """
        for bound in (region.start, region.extent, region.step):
            if isinstance(bound, Expr):
                self.visit(bound, ctx)
        if len(region.yield_values) > len(region.params):
            ctx.error(
                region,
                f"LoopRegion yields {len(region.yield_values)} values but has "
                f"{len(region.params)} params",
            )
        memo = self._region_memo(region, ctx)
        memo[id(region.induction_var)] = (region.induction_var, region.induction_var.annotation)
        inner = TypeInferVisitor(
            memo=memo,
            owns_body=self._owns_body,
            ranges=self._ranges,
        )
        body_type = inner.visit(region.body, ctx)
        carried = region.params[: len(region.yield_values)]
        for index, (phi, y) in enumerate(zip(carried, region.yield_values, strict=True)):
            entry_type = memo[id(phi)][1]
            if not types_compatible(entry_type, inner.visit(y, ctx)):
                ctx.error(
                    region,
                    f"LoopRegion yield {index} type mismatch for param {phi.name!r}",
                )
        if not carried:
            return body_type
        if len(carried) == 1:
            return inner.visit(carried[0], ctx)
        return TupleType(fields=tuple(inner.visit(phi, ctx) for phi in carried))

    def visit_MeshRegion(self, expr: MeshRegion, ctx: TypeInferContext) -> Type:
        """Type a region against the participants in force inside it.

        Entering a scope composes it onto the mesh in force, so the body reads
        the whole nesting. The resulting mesh goes down on a child context, so
        the caller's own scope survives the recursion. The region types as its
        body does: who runs the work is not a fact about the shape of what it
        produced, and what one unit costs is cost's question.
        """
        memo = self._region_memo(expr, ctx)
        mesh = make_mesh(ctx.current_mesh, expr.mesh) if ctx.current_mesh else expr.mesh
        inner = TypeInferVisitor(memo=memo, owns_body=self._owns_body, ranges=self._ranges)
        return inner.visit(expr.body, replace(ctx, current_mesh=mesh, memo=memo))

    def visit_Function(self, fn: Function, ctx: TypeInferContext) -> Type:
        """Refresh one complete function after binding its parameter types."""
        if ctx.frame is not None and ctx.frame.function is not fn:
            ctx = replace(ctx, frame=replace(ctx.frame, function=fn))
        memo = {id(param): (param, param.annotation) for param in fn.params}
        if fn.body is not None:
            TypeInferVisitor(
                memo=memo,
                owns_body=self._owns_body,
                ranges=self._ranges,
            ).visit(fn.body, replace(ctx, memo=memo))
        for nested in fn.variants:
            TypeInferVisitor(
                owns_body=self._owns_body,
                ranges=self._ranges,
            ).visit(nested, ctx)
        return callable_type_for(fn.params, fn.return_type)

    def visit_leaf_ShapeOf(self, shape_of: ShapeOf, _operands, ctx: TypeInferContext) -> Type:
        """A ``tir.ShapeOf`` always carries its own concrete (rank-0 i32) type at construction.

        A ``tir.ShapeOf`` always carries its own concrete (rank-0 i32)
        type at construction; it has no children to derive from.
        """
        return shape_of.type

    def default_visit_leaf(self, expr: Expr, _operands, ctx: TypeInferContext) -> Type:
        ctx.error(expr, f"no typeinfer rule for Expr subclass {type(expr).__name__}")


def inference_type(
    expr: Expr,
    ctx: TypeInferContext | None = None,
    *,
    ranges: bool = False,
) -> Type:
    """Infer *expr*; ``ranges=True`` refreshes value ranges but not stored types."""
    return TypeInferVisitor(owns_body=False, ranges=ranges).visit(
        expr, ctx if ctx is not None else TypeInferContext()
    )


__all__ = [
    "TypeInferVisitor",
    "inference_type",
    "input_scope_refusal",
]
