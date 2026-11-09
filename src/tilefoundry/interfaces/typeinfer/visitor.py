"""Type inference over expressions, functions, and authored regions."""

from __future__ import annotations

import typing
from dataclasses import MISSING, replace

from tilefoundry import services
from tilefoundry.interfaces import valuation
from tilefoundry.ir import core, dim, expr, types, utils
from tilefoundry.ir.types import int_tuple

from .context import TypeInferContext, TypeInferResults
from .reachability import check_input_reachability
from .registry import registry


def check_module_index(slot: core.ModuleList, index=MISSING, *, index_type=None):
    """Validate one entry call or subscript, returning its normalized index.

    Validate the slot from the current valuation.Context's module. A dynamic index
    remains an expression; runtime bounds are checked when it is evaluated.
    """
    if index is MISSING:
        if slot.indexed:
            raise ValueError(f"ModuleList slot {slot.key!r} must be indexed before calling")
        return None
    if not slot.indexed:
        raise ValueError(f"single slot {slot.key!r} cannot be indexed")
    if isinstance(index, expr.Expr) or index_type is not None:
        index_type = index.type if index_type is None else index_type

        if (
            not isinstance(index_type, types.TensorType)
            or index_type.shape != ()
            or not isinstance(index_type.dtype, expr.IntegerDType)
        ):
            raise ValueError(f"ModuleList slot {slot.key!r}: index must be an integer scalar")
    if isinstance(index, expr.Constant):
        index = index.value
    if isinstance(index, bool) or not isinstance(index, (int, expr.Expr)):
        raise ValueError(f"ModuleList slot {slot.key!r}: index must be an integer scalar")
    if isinstance(index, int):
        if not 0 <= index < len(slot):
            raise ValueError(
                f"ModuleList slot {slot.key!r}: index {index!r} out of range for N={len(slot)}"
            )
    elif any(element is not slot.elements[0] for element in slot.elements[1:]):
        raise ValueError(
            f"ModuleList slot {slot.key!r}: variable index requires the same definition"
        )
    return index


def require_axis_groups(node, type_, ctx):
    for tensor in utils.tensor_types(type_):
        if tensor.layout is not None:
            try:
                int_tuple.compatible_profile(tensor.layout.shape, tensor.shape)
            except ValueError as error:
                ctx.error(node, str(error))


def record_range(node: expr.Expr, results: TypeInferResults) -> None:
    """Replace one expression's derived range with this inference result."""
    if results.value_range is None:
        node.detach_metadata(expr.RangeMetadata)
    else:
        node.attach_metadata(expr.RangeMetadata(*results.value_range))


def target_rule(op_cls: type, ctx: TypeInferContext):
    """The rule registered for this Op under the walk's Target, if any."""
    target = ctx.resolve_target()
    if target is None:
        return None
    return registry.get((type(target), op_cls), None)


def call_function(
    call: expr.Call,
    callee: core.Function,
    arg_types: tuple[types.Type, ...],
    ctx: TypeInferContext,
    *,
    child_frame: valuation.Context | None = None,
) -> types.Type:
    child = child_frame is not None and not isinstance(call.target, core.Function)
    supplied = tuple(param for param in callee.params if not (child and param.is_const))
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
            memo[id(param)] = (param, param.type)
            continue
        index, arg_type = next(given)
        declared = param.annotation
        if not utils.types_compatible(declared, arg_type):
            ctx.error(
                call,
                f"hir Function call {callee.name!r}: arg {index} type mismatch — "
                f"callee param {param.name!r} expects {declared!r}, got {arg_type!r}",
            )
        memo[id(param)] = (param, arg_type)

    if callee.body is None or callee.variants:
        return callee.return_type

    inner = replace(child_frame or ctx, function=callee, owner=callee, parent=None, call_site=None, memo={})
    context = (
        None
        if inner.module is None
        else (id(inner.module), inner.target, inner.topologies)
    )
    key = (id(callee), arg_types, ctx.current_mesh, context)
    cached = ctx.instantiated_memo.get(key)
    if cached is not None:
        return cached
    result = TypeInferVisitor(memo=memo, owns_body=False, ranges=False).visit(
        callee.body, inner
    )
    ctx.remember_instance(key, result)
    return result


class TypeInferVisitor(valuation.Visitor[types.Type]):
    """Derive one type for each ``Expr`` kind.

    See [hir §1.1](docs/spec/hir.md#11-function) and
    [visitor-registry §4](docs/spec/visitor-registry.md#4-instance-1--typeinfer).

    The context owns one identity memo for the current inference scope. A
    missing leaf raises through ``default_visit_leaf`` rather than trusting a
    stale ``expr.type``.
    """

    def __init__(self, *, memo=None, owns_body: bool = True, ranges: bool = False) -> None:
        super().__init__(memo=memo)
        self.memo_supplied = memo is not None
        self.visit_depth = 0
        self.owns_body = owns_body
        self.ranges = ranges

    def visit(self, node: expr.Expr, ctx: TypeInferContext) -> types.Type:
        """Derive one type while preserving the active execution domain."""
        outermost = self.visit_depth == 0
        if outermost:
            if self.memo_supplied:
                ctx = replace(ctx, memo=self.memo)
            else:
                self.memo = ctx.memo
        cached = id(node) in self.memo
        self.visit_depth += 1
        try:
            results = super().visit(node, ctx)
            if not isinstance(results, TypeInferResults):
                results = TypeInferResults(results)
            result = dim.canonicalize_dims(results.type)
            if isinstance(node, expr.Call):
                require_axis_groups(node, result, ctx)
            self.memo[id(node)] = (node, result)
            if self.owns_body:
                node.retype(result)
            if self.ranges and not cached:
                record_range(node, results)
            return result
        finally:
            self.visit_depth -= 1

    def visit_operands(self, node: expr.Expr, ctx: TypeInferContext) -> tuple[types.Type, ...]:
        """Infer Expr operands while treating symbolic dimensions as meta-scalars."""
        return tuple(
            types.TensorType.from_umat_scalar() if isinstance(child, expr.DimVar) else self.visit(child, ctx)
            for child in core.expr_children(node)
        )

    def visit_leaf_Var(self, var: expr.Var, operands, ctx: TypeInferContext) -> types.Type:
        return var.type

    def visit_leaf_Constant(self, c: expr.Constant, operands, ctx: TypeInferContext) -> types.Type:
        return c.type

    def visit_leaf_Call(self, call: expr.Call, arg_types, ctx: TypeInferContext) -> types.Type:
        target = call.target
        if call.args and isinstance(call.args[0], core.ModuleList):
            slot = call.args[0]
            if ctx.module is not None:
                slot = ctx.module.children[slot.key]
            try:
                index = check_module_index(slot, call.args[1], index_type=arg_types[1])
            except ValueError as error:
                ctx.error(call, str(error))
            return arg_types[0].fields[index if isinstance(index, int) else 0]
        if isinstance(target, expr.Call):
            arg_types = arg_types[1:]
        try:
            if isinstance(target, core.ModuleList) and ctx.module is not None:
                check_module_index(ctx.module.children[target.key])
            child_frame = ctx.callee_context(call)
            if child_frame is not None:
                target = child_frame.function
        except (KeyError, ValueError) as error:
            ctx.error(call, str(error))
        try:
            check_input_reachability(target, arg_types, ctx.current_mesh)
        except ValueError as error:
            ctx.error(call, str(error))
        if isinstance(target, core.Function):
            return call_function(call, target, arg_types, ctx, child_frame=child_frame)
        op_cls = type(target)
        fn = target_rule(op_cls, ctx) or registry.get(op_cls, None)
        if fn is None:
            ctx.error(call, f"no typeinfer registered for {op_cls.__name__}")
        try:
            return fn(call, ctx)
        except expr.VerifyError:
            raise
        except ValueError as error:
            ctx.error(call, str(error))

    def visit_leaf_ModuleList(self, slot, operands, ctx: TypeInferContext) -> types.Type:
        try:
            current = ctx.module.children[slot.key] if ctx.module is not None else slot
            return current.type
        except (KeyError, ValueError) as error:
            ctx.error(slot, str(error))

    def visit_leaf_Tuple(self, tup: expr.Tuple, operands, ctx: TypeInferContext) -> types.Type:
        """Visit Tuple.

        Structural: the field types of the (possibly just-elaborated)
        elements, never the node's own stamped ``.type`` ([hir §1.1](docs/spec/hir.md#11-function)).
        """
        return types.TupleType(fields=operands)

    def region_memo(
        self, region: core.LoopRegion | core.MeshRegion, ctx: TypeInferContext
    ) -> dict[int, tuple[expr.Expr, types.Type]]:
        """Infer arguments and bind each region parameter to its argument's type.

        A region parameter has no type of its own: it is the value its argument
        takes as the region is entered, so a type stored on it by an earlier
        walk does not constrain this one. A walk that owns the body stores the
        entry type on every parameter, captures the body never reads included.
        """
        arg_types = tuple(self.visit(arg, ctx) for arg in region.args)

        valuation.verify_region_isolated(region, ctx)
        if self.owns_body:
            for param, arg_type in zip(region.params, arg_types, strict=True):
                param.restamp(param.annotation, arg_type)
        return {
            **ctx.memo,
            **{
                id(param): (param, arg_type)
                for param, arg_type in zip(region.params, arg_types, strict=True)
            },
        }

    def visit_LoopRegion(self, region: core.LoopRegion, ctx: TypeInferContext) -> types.Type:
        """Infer a loop after binding its induction and carried variables.

        The entry values give the carried types and the result, so a loop that
        runs no iteration has the type it was entered with. Each yielded value
        must fit the entry type of the parameter it carries into; a yield that
        is more specific does not change that type.
        """
        for bound in (region.start, region.extent, region.step):
            if isinstance(bound, expr.Expr):
                self.visit(bound, ctx)
        if len(region.yield_values) > len(region.params):
            ctx.error(
                region,
                f"LoopRegion yields {len(region.yield_values)} values but has "
                f"{len(region.params)} params",
            )
        memo = self.region_memo(region, ctx)
        induction_type = region.induction_var.type
        memo[id(region.induction_var)] = (region.induction_var, induction_type)
        if self.ranges:
            step = expr.static_dim_value(region.step)
            bounds = None
            if step is not None and step > 0:
                try:
                    start, stop = services.dim_range(region.start), services.dim_range(region.extent)
                    if start is not None and stop is not None and start[0] < stop[1] - 1:
                        bounds = (start[0], stop[1] - 1)
                except (NotImplementedError, ArithmeticError, TypeError, ValueError):
                    bounds = None
            record_range(
                region.induction_var,
                TypeInferResults(induction_type, value_range=bounds),
            )
        inner = TypeInferVisitor(
            memo=memo,
            owns_body=self.owns_body,
            ranges=self.ranges,
        )
        ctx = replace(ctx.enter_region(region), memo=memo)
        body_type = inner.visit(region.body, ctx)
        carried = region.params[: len(region.yield_values)]
        for index, (phi, y) in enumerate(zip(carried, region.yield_values, strict=True)):
            entry_type = memo[id(phi)][1]
            yield_type = inner.visit(y, ctx)
            if not utils.types_compatible(entry_type, yield_type):
                hint = ""
                for entry, yielded in zip(
                    utils.tensor_types(entry_type), utils.tensor_types(yield_type)
                ):
                    shards = tuple(types.shard_layout_of(t.layout) for t in (entry, yielded))
                    frame = next((shard.mesh for shard in shards if shard is not None), None)
                    states = tuple(
                        utils.attrs_by_axis(t, None if shard is not None else frame)
                        for t, shard in zip((entry, yielded), shards)
                    )
                    for key in states[0].keys() | states[1].keys():
                        before, after = (state.get(key, types.Broadcast()) for state in states)
                        if before != after:
                            labels = {types.Broadcast(): "B", types.Varying(): "V"}
                            hint = (
                                f": axis {key} enters {labels.get(before, repr(before))} "
                                f"and leaves {labels.get(after, repr(after))}"
                            )
                            if before in labels and after in labels:
                                hint += (
                                    f"; reshard the entry to {labels[after]} "
                                    f"or the yield to {labels[before]}"
                                )
                            break
                    if hint:
                        break
                ctx.error(
                    region, f"LoopRegion yield {index} type mismatch for param {phi.name!r}{hint}"
                )
        if not carried:
            return body_type
        if len(carried) == 1:
            return inner.visit(carried[0], ctx)
        return types.TupleType(fields=tuple(inner.visit(phi, ctx) for phi in carried))

    def visit_MeshRegion(self, node: core.MeshRegion, ctx: TypeInferContext) -> types.Type:
        """Type a region against the participants in force inside it.

        Entering a scope composes it onto the mesh in force, so the body reads
        the whole nesting. The resulting mesh goes down on a child context, so
        the caller's own scope survives the recursion. The region types as its
        body does: who runs the work is not a fact about the shape of what it
        produced, and what one unit costs is cost's question.
        """
        memo = self.region_memo(node, ctx)
        inner = TypeInferVisitor(
            memo=memo,
            owns_body=self.owns_body,
            ranges=self.ranges,
        )
        return inner.visit(node.body, replace(ctx.enter_region(node), memo=memo))

    def visit_Function(self, fn: core.Function, ctx: TypeInferContext) -> types.Type:
        """Refresh one complete function after binding its parameter types."""
        ctx = replace(ctx, owner=fn)
        if ctx.module is not None and ctx.function is not fn:
            ctx = replace(ctx, function=fn)
        for param in fn.params:
            require_axis_groups(param, param.annotation, ctx)
        memo = {
            id(param): ctx.memo[id(param)]
            if id(param) in ctx.memo
            else (param, types.bind_annotation(param.type))
            for param in fn.params
        }
        if self.owns_body:
            for param in fn.params:
                param.restamp(param.annotation, memo[id(param)][1])
        if fn.body is not None:
            TypeInferVisitor(
                memo=memo,
                owns_body=self.owns_body,
                ranges=self.ranges,
            ).visit(fn.body, replace(ctx, memo=memo))
        for nested in fn.variants:
            TypeInferVisitor(
                owns_body=self.owns_body,
                ranges=self.ranges,
            ).visit(nested, ctx)
        return types.callable_type_for(fn.params, fn.return_type)

    def visit_leaf_ShapeOf(self, shape_of: core.ShapeOf, operands, ctx: TypeInferContext) -> types.Type:
        """A ``tir.ShapeOf`` always carries its own concrete (rank-0 i32) type at construction.

        A ``tir.ShapeOf`` always carries its own concrete (rank-0 i32)
        type at construction; it has no children to derive from.
        """
        return shape_of.type

    @typing.override
    def default_visit_leaf(self, node: expr.Expr, operands, ctx: TypeInferContext) -> types.Type:
        ctx.error(node, f"no typeinfer rule for Expr subclass {type(node).__name__}")


def infer_function(fn: core.Function, ctx: TypeInferContext) -> core.Function:
    """Refresh a standalone function's values and inferred result signature."""
    TypeInferVisitor(ranges=True).visit(fn, ctx)
    fn.restamp_return(fn.body.type if fn.body is not None else fn.return_type)
    return fn


def inference_type(
    node: expr.Expr,
    ctx: TypeInferContext | None = None,
    *,
    ranges: bool = False,
) -> types.Type:
    """Infer *expr*; ``ranges=True`` refreshes value ranges but not stored types."""
    return TypeInferVisitor(owns_body=False, ranges=ranges).visit(
        node, ctx if ctx is not None else TypeInferContext()
    )


__all__ = ["TypeInferVisitor", "infer_function", "inference_type"]
