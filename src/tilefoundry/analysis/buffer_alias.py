"""Resolve backing buffers and prove view declarations once for memory analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from tilefoundry.ir.core import Call, Constant, Expr, Tuple
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.isl_interop import shape_to_isl_set
from tilefoundry.ir.types.utils import is_literal_shape
from tilefoundry.ir.visitor import ExprVisitor
from tilefoundry.visitor_registry.access_relation import renaming_relation
from tilefoundry.visitor_registry.buffer_alias import aliased_operand

from .errors import AnalysisError
from .iteration_scope import walk_scopes
from .metadata import BufferAliasMetadata
from .visitor import AnalyzeContext


def storage_source(
    value: Expr, bindings: Mapping[int, Expr], roots: Mapping[int, Expr]
) -> Expr | None:
    """One storage-sharing edge, using already resolved roots for tuple projections."""
    bound = bindings.get(id(value))
    if bound is not None:
        return bound
    if isinstance(value, MeshRegion):
        return value.body
    if isinstance(value, LoopRegion):
        if not value.yield_values:
            return value.body
        return value.params[0] if len(value.yield_values) == 1 else None
    if isinstance(value, Call):
        if isinstance(value.target, TupleGetItem):
            index = value.args[1]
            if isinstance(index, Constant) and type(index.value) is int:
                source = roots[id(value.args[0])]
                if isinstance(source, Tuple) and 0 <= index.value < len(source.elements):
                    return source.elements[index.value]
                if isinstance(source, LoopRegion) and 0 <= index.value < len(source.yield_values):
                    return source.params[index.value]
        position = aliased_operand(value)
        if position is not None:
            return value.args[position]
    return None


class BufferAliasVisitor(ExprVisitor[None]):
    """Collect definition-ordered roots and check each logical view once."""

    def __init__(self, function: Function, context: AnalyzeContext) -> None:
        super().__init__(root_function=function)
        self.logical = replace(context, topology_level=None)
        self.declarations = {
            key: scope for scope in walk_scopes(context.root) for key in scope.relations
        }
        self.roots: dict[int, Expr] = {}
        self.bindings: dict[int, Expr] = {}

    def default_visit_leaf(self, value: Expr, operands: tuple[None, ...], ctx=None) -> None:
        del operands, ctx
        source = storage_source(value, self.bindings, self.roots)
        if isinstance(value, Call) and (position := aliased_operand(value)) is not None:
            operand = value.args[position]
            relation = renaming_relation(
                value,
                self.logical,
                self.declarations[id(value)].projected_relations(value, self.logical),
            ).relation
            box = (
                shape_to_isl_set(tuple(operand.type.shape), {})
                if is_literal_shape(operand.type.shape)
                else None
            )
            if (
                box is None
                or not relation.is_single_valued()
                or not relation.is_injective()
                or not relation.range().is_subset(box)
            ):
                param = tuple(
                    param for param in type(value.target)._op_schema.signature
                    if param.kind == "input"
                )[position]
                raise AnalysisError(
                    f"{type(value.target).__name__} declares its result is {param.name}'s bytes, "
                    f"but its access relation {relation} is not single-valued, injective, "
                    "and within the operand"
                )
        self.roots[id(value)] = value if source is None else self.roots[id(source)]

    def visit_MeshRegion(self, region: MeshRegion, ctx=None) -> None:
        for argument in region.args:
            self.visit(argument, ctx)
        self.bindings.update((id(param), arg) for param, arg in region.captures())
        for parameter in region.params:
            self.visit(parameter, ctx)
        self.visit(region.body, ctx)
        self.default_visit_leaf(region, (), ctx)

    def visit_LoopRegion(self, region: LoopRegion, ctx=None) -> None:
        for argument in region.args:
            self.visit(argument, ctx)
        self.bindings.update((id(param), arg) for param, arg in region.captures())
        self.visit(region.induction_var, ctx)
        for parameter in region.params:
            self.visit(parameter, ctx)
        self.visit(region.body, ctx)
        for yielded in region.yield_values:
            self.visit(yielded, ctx)
        self.default_visit_leaf(region, (), ctx)


def analyze_buffer_alias(function: Function, context: AnalyzeContext) -> BufferAliasMetadata:
    """The first memory stage: resolve each storage edge in definition order."""
    visitor = BufferAliasVisitor(function, context)
    for parameter in function.params:
        visitor.visit(parameter)
    visitor.visit_function_body(function)
    return BufferAliasMetadata(visitor.roots, visitor.bindings)
