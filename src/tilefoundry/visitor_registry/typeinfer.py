"""Type inference over expressions, functions, and authored regions."""

from __future__ import annotations

from dataclasses import replace

from tilefoundry.ir.core.expr import Call, Constant, Expr, Tuple, Var
from tilefoundry.ir.core.metadata import (
    RangeMetadata,
    attach_metadata,
    detach_metadata,
)
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.sharding.reshard import Reshard as HirReshard
from tilefoundry.ir.mesh_scope import covered_by_scope, storage_reaches
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.types.callable_type import callable_type_for
from tilefoundry.ir.types.mesh import make_mesh
from tilefoundry.ir.types.shard_layout import ShardLayout
from tilefoundry.ir.types.substitute import canonicalize_dims
from tilefoundry.ir.types.tensor_type import TupleType, Type
from tilefoundry.ir.types.utils import types_compatible
from tilefoundry.ir.visitor import ExprVisitor

from .contexts import FunctionScope, TypeInferContext, TypeInferResults
from .registries import typeinfer_registry


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
        if ctx.current_mesh is not None and not isinstance(target, HirReshard):
            for index, arg_type in enumerate(arg_types):
                layout = getattr(arg_type, "layout", None)
                if not isinstance(layout, ShardLayout):
                    continue
                if not covered_by_scope(layout.mesh, ctx.current_mesh):
                    ctx.error(
                        call,
                        f"input {index} is laid out more finely than the scope it "
                        "runs in; write it inside that scope, or reshard it back first",
                    )
                if not storage_reaches(arg_type.storage, layout.mesh, ctx.current_mesh):
                    ctx.error(
                        call,
                        f"input {index} is laid out more coarsely and kept in "
                        f"{arg_type.storage.name.lower()}, which does not reach the units "
                        "this runs on; reshard it to smem or gmem first",
                    )
        if isinstance(target, Function):
            return self._call_function(call, target, arg_types, ctx)
        op_cls = type(target)
        fn = typeinfer_registry.lookup(op_cls)
        if fn is None:
            ctx.error(call, f"no typeinfer registered for {op_cls.__name__}")
        return fn(call, ctx)

    def _call_function(
        self,
        call: Call,
        callee: Function,
        arg_types: tuple[Type, ...],
        ctx: TypeInferContext,
    ) -> Type:
        child = ctx.child_for(callee)
        supplied = tuple(
            param for param in callee.params if not (child is not None and param.is_const)
        )
        if len(arg_types) != len(supplied):
            kind = "activation(s)" if child is not None else "parameter(s)"
            ctx.error(
                call,
                f"hir Function call {callee.name!r}: arity mismatch — "
                f"callee declares {len(supplied)} {kind}, call passed {len(arg_types)}",
            )

        given = iter(enumerate(arg_types))
        memo = {}
        for param in callee.params:
            if child is not None and param.is_const:
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

        key = (id(callee), arg_types)
        cached = ctx.instantiated_memo.get(key)
        if cached is not None:
            return cached
        result = TypeInferVisitor(memo=memo, owns_body=False, ranges=False).visit(
            callee.body, ctx.for_callee(callee)
        )
        ctx.instantiated_memo[key] = result
        return result

    def visit_leaf_Tuple(self, tup: Tuple, operands, ctx: TypeInferContext) -> Type:
        """Visit Tuple.

        Structural: the field types of the (possibly just-elaborated)
        elements, never the node's own stamped ``.type`` ([hir §1.1](docs/spec/hir.md#11-function)).
        """
        return TupleType(fields=operands)

    def visit_LoopRegion(self, region: LoopRegion, ctx: TypeInferContext) -> Type:
        """Infer a loop after binding its induction and carried variables."""
        inits = tuple(self.visit(arg, ctx) for arg in region.init_args)
        memo = {
            **self._memo,
            id(region.induction_var): (region.induction_var, region.induction_var.annotation),
            **{id(phi): (phi, type_) for phi, type_ in zip(region.carried_args, inits)},
        }
        inner = TypeInferVisitor(
            memo=memo,
            owns_body=self._owns_body,
            ranges=self._ranges,
        )
        body_type = inner.visit(region.body, ctx)
        for y in region.yield_values:
            inner.visit(y, ctx)
        if not region.carried_args:
            return body_type
        if len(region.carried_args) == 1:
            return inner.visit(region.carried_args[0], ctx)
        return TupleType(fields=tuple(inner.visit(phi, ctx) for phi in region.carried_args))

    def visit_MeshRegion(self, expr: MeshRegion, ctx: TypeInferContext) -> Type:
        """Type a region against the participants in force inside it.

        Entering a scope composes it onto the mesh in force, so the body reads
        the whole nesting. The resulting mesh goes down on a child context, so
        the caller's own scope survives the recursion. The region types as its
        body does: who runs the work is not a fact about the shape of what it
        produced, and what one unit costs is cost's question.
        """
        arg_types = tuple(self.visit(arg, ctx) for arg in expr.args)
        if len(arg_types) != len(expr.params):
            ctx.error(
                expr,
                f"mesh scope expects {len(expr.params)} argument(s), got {len(arg_types)}",
            )
        for index, (param, arg_type) in enumerate(zip(expr.params, arg_types, strict=True)):
            if not types_compatible(param.annotation, arg_type):
                ctx.error(
                    expr,
                    f"mesh scope arg {index} type mismatch for param {param.name!r}",
                )
        memo = {
            **ctx.memo,
            **{
                id(param): (param, arg_type)
                for param, arg_type in zip(expr.params, arg_types, strict=True)
            },
        }
        from tilefoundry.ir.hir.verify import _verify_isolated  # noqa: PLC0415

        _verify_isolated(expr, ctx)
        mesh = make_mesh(ctx.current_mesh, expr.mesh) if ctx.current_mesh else expr.mesh
        return self.visit(expr.body, replace(ctx, current_mesh=mesh, memo=memo))

    def visit_Function(self, fn: Function, ctx: TypeInferContext) -> Type:
        """Refresh one complete function after binding its parameter types."""
        if ctx.scope is not None and ctx.scope.function is not fn:
            ctx = replace(ctx, scope=FunctionScope(ctx.scope.module, fn))
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


__all__ = ["TypeInferVisitor", "inference_type"]
