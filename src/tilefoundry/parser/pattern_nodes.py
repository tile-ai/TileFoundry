"""Concrete executable AST Pattern nodes.

Pattern matching and construction live together here; shared parser state,
rules, and runtime helpers remain in ast_pattern.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, get_args, get_origin

from tilefoundry.ir.constraints import (
    ConstraintProvenance,
    LayoutConstraint,
    MeshConstraint,
    ScheduleConstraintMetadata,
    SourceLocation,
    StorageConstraint,
)
from tilefoundry.ir.constraints.layout import _LAYOUT_WILDCARD
from tilefoundry.ir.core import (
    BindingMetadata,
    ExecutionDomainMetadata,
    get_metadata,
    replace_metadata,
)
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.tir.launch import launch_call
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import Broadcast, Layout, Partial, Split

from .ast_pattern import (
    _BINARY_OPERATORS,
    _RETURN_TYPE,
    _TYPE_INFER_CONTEXT,
    _UNARY_OPERATORS,
    AstChild,
    AstMatch,
    AstNodePattern,
    AstPattern,
    AstRule,
    BindPattern,
    BranchPattern,
    CanonicalDTypeRule,
    CapturePattern,
    ChildPattern,
    ChoicePattern,
    ConditionPattern,
    ElementPattern,
    FieldPattern,
    FuncParserContext,
    FunctionRole,
    LayoutPositionRule,
    LayoutShapeRule,
    LazyPattern,
    LiteralPattern,
    LoopFrame,
    MatchContext,
    MatchFailure,
    OptionalPattern,
    ParseError,
    PatternFailure,
    PredicatePattern,
    ReferencePattern,
    RepeatPattern,
    SequencePattern,
    ShapeDimRule,
    ShapeTupleRule,
    StorageValueRule,
    TensorLayoutStorageRule,
    TensorPositionRule,
    _constant,
    _infer_call,
    _resolve_reference,
    _slice_size,
    attach_authored_metadata,
    parse_node,
    runtime,
)


class DimExprPattern(ElementPattern):
    element_name = "dim_expr"
    syntax = LazyPattern(
        lambda: AstNodePattern(
            ast.expr,
            ChoicePattern(
                BranchPattern(
                    "dim_literal",
                    AstNodePattern(
                        ast.Constant,
                        PredicatePattern(
                            "integer-literal",
                            lambda node, context: (
                                isinstance(node.value, int) and not isinstance(node.value, bool)
                            ),
                        ),
                        CapturePattern("value", lambda node, context: node.value),
                    ),
                    pattern_id="dim.literal",
                ),
                BranchPattern(
                    "dim_name",
                    AstNodePattern(
                        ast.Name,
                        FieldPattern("id", CapturePattern("name", lambda value, context: value)),
                    ),
                    pattern_id="dim.name",
                ),
                BranchPattern(
                    "dim_reference",
                    AstNodePattern(ast.Attribute),
                    pattern_id="dim.reference",
                ),
                BranchPattern(
                    "dim_binary",
                    AstNodePattern(
                        ast.BinOp,
                        FieldPattern(
                            "op",
                            ChoicePattern(
                                AstNodePattern(ast.Add),
                                AstNodePattern(ast.Sub),
                                AstNodePattern(ast.Mult),
                                AstNodePattern(ast.FloorDiv),
                                AstNodePattern(ast.Mod),
                            ),
                        ),
                        FieldPattern(
                            "left",
                            ChildPattern("left", lambda: DimExprPattern(), "dim_expr", "left"),
                        ),
                        FieldPattern(
                            "right",
                            ChildPattern("right", lambda: DimExprPattern(), "dim_expr", "right"),
                        ),
                        CapturePattern(
                            "operator",
                            lambda node, context: _BINARY_OPERATORS[type(node.op)],
                        ),
                    ),
                    pattern_id="dim.binary",
                ),
                BranchPattern(
                    "dim_call",
                    AstNodePattern(
                        ast.Call,
                        FieldPattern(
                            "func",
                            ChoicePattern(
                                AstNodePattern(
                                    ast.Name,
                                    FieldPattern(
                                        "id",
                                        CapturePattern("callee", lambda value, context: value),
                                    ),
                                ),
                                AstNodePattern(ast.Attribute),
                            ),
                        ),
                        FieldPattern(
                            "args",
                            RepeatPattern(
                                ChildPattern(
                                    "arg_{index}",
                                    lambda: DimExprPattern(),
                                    "dim_expr",
                                    "argument",
                                )
                            ),
                        ),
                        FieldPattern("keywords", SequencePattern()),
                    ),
                    pattern_id="dim.call",
                ),
            ),
        )
    )

    @staticmethod
    def construct(match, children, context):
        if match.branch_id == "dim_literal":
            return match.captures["value"]
        if match.branch_id == "dim_name":
            return _resolve_reference(match.node, context)
        if match.branch_id == "dim_reference":
            return _resolve_reference(match.node, context)
        if match.branch_id == "dim_binary":
            try:
                return match.captures["operator"](children["left"], children["right"])
            except (TypeError, ValueError, ZeroDivisionError) as error:
                raise ParseError.from_node(match.node, context, str(error)) from error
        if match.branch_id == "dim_call":
            callee = _resolve_reference(match.node.func, context)
            args = tuple(value for name, value in children.items() if name.startswith("arg_"))
            if not callable(callee):
                raise ParseError.from_node(
                    match.node, context, "dimension call target is not callable"
                )
            try:
                return callee(*args)
            except (TypeError, ValueError) as error:
                raise ParseError.from_node(match.node, context, str(error)) from error
        raise RuntimeError(f"no constructor branch for {match.branch_id!r}")

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (ShapeDimRule(),)


class ShapePattern(ElementPattern):
    element_name = "shape"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            BranchPattern(
                "tuple_children",
                AstNodePattern(
                    ast.Tuple,
                    FieldPattern("ctx", AstNodePattern(ast.Load)),
                    PredicatePattern(
                        "shape-tuple",
                        lambda node, context: (
                            not any(isinstance(item, ast.Tuple) for item in node.elts)
                        ),
                    ),
                    FieldPattern(
                        "elts",
                        RepeatPattern(
                            ChildPattern(
                                "dim_{index}",
                                lambda: DimExprPattern(),
                                "tensor_dim_expr",
                                "dim_expr",
                            )
                        ),
                    ),
                ),
                pattern_id="tensor.shape",
            ),
            BranchPattern(
                "shape_reference",
                ChoicePattern(AstNodePattern(ast.Name), AstNodePattern(ast.Attribute)),
                pattern_id="tensor.shape.reference",
            ),
        )
    )

    @staticmethod
    def construct(match, children, context):
        if match.branch_id == "shape_reference":
            return _resolve_reference(match.node, context)
        return tuple(children.values())

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (ShapeTupleRule(),)


class DTypePattern(ElementPattern):
    element_name = "dtype"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            BranchPattern(
                "dtype_literal",
                AstNodePattern(
                    ast.Constant,
                    FieldPattern("value", LiteralPattern(value_type=str)),
                ),
                pattern_id="tensor.dtype.literal",
            ),
            BranchPattern(
                "dtype_reference",
                ReferencePattern(),
                pattern_id="tensor.dtype.reference",
            ),
        )
    )

    @staticmethod
    def construct(match, children, context):
        if match.branch_id == "dtype_literal":
            try:
                return runtime.DType.from_name(match.node.value)
            except ValueError as error:
                raise ParseError.from_node(match.node, context, str(error)) from error
        elif match.branch_id == "dtype_reference":
            if isinstance(match.node, ast.Name) and match.node.id in runtime.DType._members():
                return runtime.DType.from_name(match.node.id)
            value = _resolve_reference(match.node, context)
            if isinstance(value, str):
                try:
                    return runtime.DType.from_name(value)
                except ValueError as error:
                    raise ParseError.from_node(match.node, context, str(error)) from error
            return value
        raise RuntimeError(f"no constructor branch for {match.branch_id!r}")

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (CanonicalDTypeRule(),)


class ExplicitLayoutPattern(ElementPattern):
    element_name = "explicit_layout"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "explicit_layout",
            AstNodePattern(
                ast.Tuple,
                FieldPattern(
                    "elts",
                    SequencePattern(
                        AstNodePattern(
                            ast.Tuple,
                            ChoicePattern(
                                ConditionPattern(
                                    "active Mesh",
                                    lambda node, context: (
                                        context.function is not None
                                        and bool(context.function.state.mesh_stack)
                                    ),
                                    ChildPattern(
                                        "shape",
                                        lambda: TensorShapeLayoutPattern(),
                                        "layout_shape",
                                        "layout_shape",
                                    ),
                                ),
                                ChildPattern(
                                    "shape",
                                    lambda: ShapePattern(),
                                    "layout_shape",
                                    "layout_shape",
                                ),
                            ),
                        ),
                        AstNodePattern(
                            ast.Tuple,
                            ChildPattern(
                                "strides",
                                lambda: ShapePattern(),
                                "layout_strides",
                                "layout_strides",
                            ),
                        ),
                    ),
                ),
            ),
            pattern_id="tensor.layout.explicit",
        )
    )

    @staticmethod
    def construct(match, children, context):
        shape_or_layout = children["shape"]
        strides = children["strides"]
        if isinstance(shape_or_layout, runtime.ShardLayout):
            shape = shape_or_layout.layout.shape
        else:
            shape = shape_or_layout
        if len(shape) != len(strides):
            raise ParseError.from_node(match.node, context, "layout shape/stride rank mismatch")
        layout = runtime.Layout(shape=shape, strides=strides)
        if isinstance(shape_or_layout, runtime.ShardLayout):
            return runtime.ShardLayout(
                layout=layout,
                attrs=shape_or_layout.attrs,
                mesh=shape_or_layout.mesh,
            )
        return layout

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        LayoutShapeRule(),
        LayoutPositionRule(),
    )


class PlainLayoutPattern(ElementPattern):
    element_name = "plain_layout"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "plain_layout",
            AstNodePattern(
                ast.Tuple,
                FieldPattern(
                    "elts",
                    RepeatPattern(
                        AstNodePattern(
                            ast.expr,
                            PredicatePattern(
                                "layout-extent",
                                lambda node, context: not isinstance(node, (ast.Tuple, ast.Slice)),
                            ),
                            ChildPattern(
                                "extent_{index}",
                                lambda: DimExprPattern(),
                                "layout_extent",
                                "layout_extent",
                            ),
                        )
                    ),
                ),
            ),
            pattern_id="tensor.layout.literal",
        )
    )

    @staticmethod
    def construct(match, children, context):
        shape = tuple(children.values())
        layout = runtime.Layout(
            shape=shape,
            strides=runtime.c_order_strides(shape, mul=operator.mul),
        )
        if (
            context.situation != "mesh_layout"
            and context.function is not None
            and context.function.state.mesh_stack
        ):
            mesh = context.function.state.mesh_stack[-1]
            return runtime.ShardLayout(
                layout=layout,
                attrs=tuple(runtime.Broadcast() for _ in mesh.layout.shape),
                mesh=mesh,
            )
        return layout

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        LayoutShapeRule(),
        LayoutPositionRule(),
    )


class MeshAxisPattern(ElementPattern):
    element_name = "mesh_axis"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "mesh_axis",
            ChoicePattern(
                AstNodePattern(ast.Name),
                AstNodePattern(
                    ast.Attribute,
                    FieldPattern("value", AstNodePattern(ast.Name)),
                ),
            ),
            pattern_id="tensor.layout.mesh_axis",
        )
    )

    @staticmethod
    def construct(match, children, context):
        node = match.node
        if isinstance(node, ast.Name):
            binding = node.id
            axis_name = None
        else:
            binding = node.value.id
            axis_name = node.attr
        mesh = context.lexical_scope.lookup(binding)
        if mesh is None:
            try:
                mesh = _resolve_reference(node.value, context)
            except ParseError:
                mesh = None
        if not isinstance(mesh, runtime.Mesh):
            raise ParseError.from_node(node, context, f"{binding!r} is not an active Mesh")
        if axis_name is None:
            if len(mesh.layout.shape) != 1:
                raise ParseError.from_node(
                    node, context, "bare Mesh placement requires a one-axis mesh"
                )
            return mesh, 0
        try:
            axis = mesh.names.index(axis_name)
        except ValueError as error:
            raise ParseError.from_node(
                node, context, f"Mesh {binding!r} has no axis {axis_name!r}"
            ) from error
        return mesh, axis

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class PlacedLayoutPattern(ElementPattern):
    element_name = "placed_layout"
    syntax = LazyPattern(
        lambda: BindPattern(
            AstNodePattern(
                ast.Tuple,
                FieldPattern(
                    "elts",
                    RepeatPattern(
                        ChoicePattern(
                            AstNodePattern(
                                ast.BinOp,
                                FieldPattern("op", AstNodePattern(ast.MatMult)),
                                FieldPattern("left", AstNodePattern(ast.expr)),
                                FieldPattern(
                                    "right",
                                    ChoicePattern(
                                        AstNodePattern(
                                            ast.Tuple,
                                            FieldPattern(
                                                "elts",
                                                RepeatPattern(
                                                    MeshAxisPattern(),
                                                    minimum=1,
                                                ),
                                            ),
                                        ),
                                        MeshAxisPattern(),
                                    ),
                                ),
                            ),
                            DimExprPattern(),
                        )
                    ),
                ),
            ),
            PlacedLayoutPattern._bind,
        )
    )

    @staticmethod
    def _placement_parts(node: ast.AST) -> tuple[ast.AST, tuple[ast.AST, ...]] | None:
        """Flatten ``extent @ axis @ axis`` into one extent and its axes."""
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.MatMult):
            return None
        left = PlacedLayoutPattern._placement_parts(node.left)
        if left is None:
            extent = node.left
            axes: tuple[ast.AST, ...] = ()
        else:
            extent, axes = left
        right_axes = tuple(node.right.elts) if isinstance(node.right, ast.Tuple) else (node.right,)
        return extent, (*axes, *right_axes)

    @staticmethod
    def _bind(node: object, context: MatchContext, matched: AstMatch[Any]) -> AstMatch[Any] | None:
        assert isinstance(node, ast.Tuple)
        children: list[AstChild] = []
        bindings: list[tuple[str, int]] = []
        found_placement = False
        for tensor_axis, item in enumerate(node.elts):
            placement = PlacedLayoutPattern._placement_parts(item)
            extent_node = item
            axis_nodes: tuple[ast.AST, ...] = ()
            if placement is not None:
                found_placement = True
                extent_node, axis_nodes = placement
            extent_context = context.child(situation="layout_extent", role="layout_extent")
            if DimExprPattern().match(extent_node, extent_context) is None:
                return None
            children.append(
                AstChild(
                    f"extent_{tensor_axis}",
                    DimExprPattern(),
                    extent_node,
                    "layout_extent",
                    "layout_extent",
                )
            )
            for mesh_axis, axis_node in enumerate(axis_nodes):
                axis_context = context.child(situation="mesh_axis", role="mesh_axis")
                if MeshAxisPattern().match(axis_node, axis_context) is None:
                    return None
                child_name = f"binding_{tensor_axis}_{mesh_axis}"
                bindings.append((child_name, tensor_axis))
                children.append(
                    AstChild(
                        child_name,
                        MeshAxisPattern(),
                        axis_node,
                        "mesh_axis",
                        "mesh_axis",
                    )
                )
        if not found_placement:
            return None
        return dataclasses.replace(
            matched,
            pattern_id="tensor.layout.placed",
            branch_id="placed_layout",
            captures={
                **matched.captures,
                "rank": len(node.elts),
                "bindings": tuple(bindings),
            },
            children=tuple(children),
        )

    @staticmethod
    def construct(match, children, context):
        rank = match.captures["rank"]
        shape = tuple(children[f"extent_{axis}"] for axis in range(rank))
        bindings = tuple(
            (*children[child_name], tensor_axis)
            for child_name, tensor_axis in match.captures["bindings"]
        )
        referenced_ids = {id(mesh) for mesh, _, _ in bindings}
        if context.function is None:
            raise ParseError.from_node(
                match.node, context, "placed layout requires function context"
            )
        meshes = tuple(
            mesh for mesh in context.function.state.mesh_stack if id(mesh) in referenced_ids
        )
        if len(meshes) != len(referenced_ids):
            meshes = tuple(dict.fromkeys(mesh for mesh, _, _ in bindings))
            if len(meshes) != len(referenced_ids):
                raise ParseError.from_node(
                    match.node, context, "placement references an inactive Mesh"
                )
        mesh = meshes[0] if len(meshes) == 1 else runtime.composed(meshes)
        source_offsets: dict[int, int] = {}
        offset = 0
        for source in meshes:
            source_offsets[id(source)] = offset
            offset += len(source.layout.shape)
        attrs: list[object] = [runtime.Broadcast() for _ in mesh.layout.shape]
        for source, source_axis, tensor_axis in bindings:
            target_axis = source_offsets[id(source)] + source_axis
            if not isinstance(attrs[target_axis], runtime.Broadcast):
                raise ParseError.from_node(match.node, context, "mesh axis is bound more than once")
            attrs[target_axis] = runtime.Split(tensor_axis)
        try:
            canonical = runtime.canonical_shard_layout(shape, mesh, tuple(attrs))
            return runtime.ShardLayout(
                layout=runtime.Layout(shape=canonical.layout.shape, strides=None),
                attrs=canonical.attrs,
                mesh=canonical.mesh,
            )
        except (TypeError, ValueError) as error:
            raise ParseError.from_node(match.node, context, str(error)) from error

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        LayoutShapeRule(),
        LayoutPositionRule(),
    )


class LayoutPattern(ElementPattern):
    element_name = "layout"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            BranchPattern(
                "none",
                AstNodePattern(
                    ast.Constant,
                    FieldPattern("value", LiteralPattern(None)),
                ),
                pattern_id="tensor.layout.none",
            ),
            BranchPattern(
                "layout_reference",
                ReferencePattern(),
                pattern_id="tensor.layout.reference",
            ),
            BranchPattern(
                "identity",
                ConditionPattern(
                    "layout call",
                    lambda node, context: isinstance(node, ast.Call),
                    ChildPattern(
                        "value",
                        lambda: StaticCallPattern(),
                        "static_layout",
                        "layout",
                    ),
                ),
                pattern_id="tensor.layout.call",
            ),
            ExplicitLayoutPattern(),
            PlacedLayoutPattern(),
            PlainLayoutPattern(),
        )
    )

    @staticmethod
    def construct(match, children, context):
        if match.branch_id == "none":
            return None
        elif match.branch_id == "layout_reference":
            return context.resolve_static(match.node, runtime.LayoutBase)
        elif match.branch_id == "identity":
            return children["value"]
        raise RuntimeError(f"no constructor branch for {match.branch_id!r}")

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        LayoutShapeRule(),
        LayoutPositionRule(),
    )


class StoragePattern(ElementPattern):
    element_name = "storage"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            BranchPattern(
                "storage",
                AstNodePattern(
                    ast.Constant,
                    FieldPattern("value", LiteralPattern(value_type=str)),
                ),
                pattern_id="tensor.storage.literal",
            ),
            BranchPattern(
                "storage",
                ReferencePattern(),
                pattern_id="tensor.storage.reference",
            ),
        )
    )

    @staticmethod
    def construct(match, children, context):
        if isinstance(match.node, ast.Name) and match.node.id in {
            str(k) for k in runtime.StorageKind
        }:
            raw = match.node.id
        elif isinstance(match.node, ast.Constant):
            raw = match.node.value
        else:
            raw = _resolve_reference(match.node, context)
        try:
            return runtime.resolve_storage(raw)
        except (TypeError, ValueError) as error:
            raise ParseError.from_node(match.node, context, str(error)) from error

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (StorageValueRule(),)


class TensorOptionalSlotPattern(ElementPattern):
    element_name = "tensor_optional_slot"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            ConditionPattern(
                "role == layout",
                lambda node, context: context.role == "layout",
                LayoutPattern(),
            ),
            ConditionPattern(
                "role == storage",
                lambda node, context: context.role == "storage",
                StoragePattern(),
            ),
            ConditionPattern(
                "role == layout_or_storage and value is layout",
                lambda node, context: (
                    context.role == "layout_or_storage"
                    and TensorOptionalSlotPattern._slot_kind(node, context) == "layout"
                ),
                LayoutPattern(),
            ),
            ConditionPattern(
                "role == layout_or_storage and value is storage",
                lambda node, context: (
                    context.role == "layout_or_storage"
                    and TensorOptionalSlotPattern._slot_kind(node, context) == "storage"
                ),
                StoragePattern(),
            ),
        )
    )

    @staticmethod
    def _slot_kind(node: object, context: MatchContext) -> str | None:
        if isinstance(node, ast.Constant):
            if node.value is None:
                return "layout"
            if isinstance(node.value, str):
                return "storage"
        if isinstance(node, (ast.Tuple, ast.Call)):
            return "layout"
        if isinstance(node, (ast.Name, ast.Attribute)):
            try:
                value = _resolve_reference(node, context)
            except ParseError:
                if isinstance(node, ast.Name) and node.id in {
                    str(item) for item in runtime.StorageKind
                }:
                    return "storage"
                raise
            if isinstance(value, runtime.StorageKind):
                return "storage"
            if isinstance(value, runtime.LayoutBase):
                return "layout"
        return None

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class TensorShapeLayoutPattern(ElementPattern):
    element_name = "tensor_shape_layout"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            PlacedLayoutPattern(),
            ShapePattern(),
        )
    )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class TensorPattern(ElementPattern):
    element_name = "tensor"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "tensor",
            AstNodePattern(
                ast.Subscript,
                FieldPattern(
                    "value",
                    AstNodePattern(
                        ast.expr,
                        PredicatePattern(
                            "tensor-head",
                            lambda node, context: (
                                TensorPattern._head(node) in {"Tensor", "ConstTensor"}
                            ),
                        ),
                        CapturePattern(
                            "head",
                            lambda node, context: TensorPattern._head(node),
                        ),
                    ),
                ),
                FieldPattern(
                    "slice",
                    AstNodePattern(
                        ast.Tuple,
                        CapturePattern("slot_count", lambda node, context: len(node.elts)),
                        FieldPattern(
                            "elts",
                            ChoicePattern(
                                SequencePattern(
                                    ChildPattern(
                                        "shape_or_layout",
                                        lambda: TensorShapeLayoutPattern(),
                                        "tensor_shape",
                                        "tensor_shape_or_layout",
                                    ),
                                    ChildPattern(
                                        "dtype",
                                        lambda: DTypePattern(),
                                        "tensor_dtype",
                                        "dtype",
                                    ),
                                ),
                                SequencePattern(
                                    ChildPattern(
                                        "shape_or_layout",
                                        lambda: TensorShapeLayoutPattern(),
                                        "tensor_shape",
                                        "tensor_shape_or_layout",
                                    ),
                                    ChildPattern(
                                        "dtype",
                                        lambda: DTypePattern(),
                                        "tensor_dtype",
                                        "dtype",
                                    ),
                                    ChildPattern(
                                        "optional_0",
                                        lambda: TensorOptionalSlotPattern(),
                                        "tensor_optional_slot",
                                        "layout_or_storage",
                                    ),
                                ),
                                SequencePattern(
                                    ChildPattern(
                                        "shape_or_layout",
                                        lambda: TensorShapeLayoutPattern(),
                                        "tensor_shape",
                                        "tensor_shape_or_layout",
                                    ),
                                    ChildPattern(
                                        "dtype",
                                        lambda: DTypePattern(),
                                        "tensor_dtype",
                                        "dtype",
                                    ),
                                    ChildPattern(
                                        "optional_0",
                                        lambda: TensorOptionalSlotPattern(),
                                        "tensor_optional_slot",
                                        "layout",
                                    ),
                                    ChildPattern(
                                        "optional_1",
                                        lambda: TensorOptionalSlotPattern(),
                                        "tensor_optional_slot",
                                        "storage",
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            pattern_id="tensor.annotation",
        )
    )

    @staticmethod
    def construct(match, children, context):
        shape_or_layout = children["shape_or_layout"]
        if isinstance(shape_or_layout, runtime.LayoutBase):
            shape = shape_or_layout.shape
            layout = shape_or_layout
        else:
            shape = shape_or_layout
            layout = None
        storage = runtime.StorageKind.GMEM
        third = children.get("optional_0")
        fourth = children.get("optional_1")
        if isinstance(third, runtime.StorageKind):
            storage = third
        elif third is None or isinstance(third, runtime.LayoutBase):
            layout = third
        else:
            raise ParseError.from_node(
                match.node, context, "third Tensor slot is not layout/storage"
            )
        if fourth is not None:
            storage = fourth
        return runtime.TensorType(
            shape=shape, dtype=children["dtype"], layout=layout, storage=storage
        )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        TensorLayoutStorageRule(),
        TensorPositionRule(),
    )

    @staticmethod
    def _head(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None


class ScalarTypePattern(ElementPattern):
    element_name = "scalar_type"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "type_reference",
            ReferencePattern(),
            pattern_id="type.reference",
        )
    )

    @staticmethod
    def construct(match, children, context):
        value = _resolve_reference(match.node, context)
        if not isinstance(value, (runtime.TensorType, runtime.TupleType, runtime.UnitType)):
            raise ParseError.from_node(match.node, context, "annotation did not resolve to IR Type")
        return value

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class TypeAnnotationPattern(ElementPattern):
    element_name = "type_annotation"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            TensorPattern(),
            ScalarTypePattern(),
        )
    )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


def _constraint_value(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    raise ValueError(
        f"where layout extent must be a literal or symbolic name, got {type(node).__name__}"
    )


def _parse_partial_constraint(node: ast.AST):
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError('Partial binding must use P("reduction")')
    if node.func.id != "P" or len(node.args) != 1 or node.keywords:
        raise ValueError('Partial binding must use P("reduction")')
    value = _constraint_value(node.args[0])
    if not isinstance(value, str) or not value:
        raise ValueError("partial reduction must be a non-empty string")
    return Partial(value)


def _parse_constraint_bindings(node: ast.AST):
    if not isinstance(node, ast.Set):
        raise ValueError("layout bindings must be a set")
    bindings = []
    for item in node.elts:
        if not isinstance(item, ast.BinOp) or not isinstance(item.op, ast.MatMult):
            raise ValueError("layout bindings must use `topology @ B()` or `P()`")
        topology = _constraint_value(item.left)
        if not isinstance(topology, str) or not topology:
            raise ValueError("layout binding topology must be symbolic")
        if (
            isinstance(item.right, ast.Call)
            and isinstance(item.right.func, ast.Name)
            and item.right.func.id == "B"
            and not item.right.args
            and not item.right.keywords
        ):
            attribute = Broadcast()
        else:
            attribute = _parse_partial_constraint(item.right)
        bindings.append((topology, attribute))
    return bindings


def _where_static(node: ast.AST, context: MatchContext):
    if isinstance(node, ast.Constant):
        return node.value
    return _resolve_reference(node, context)


def _parse_layout_constraint(node: ast.AST, context: MatchContext):
    if not isinstance(node, ast.Tuple):
        raise ValueError("layout constraint must be a tuple")
    dims_node = node
    extras: tuple[ast.AST, ...] = ()
    if node.elts and isinstance(node.elts[0], ast.Tuple):
        dims_node = node.elts[0]
        extras = tuple(node.elts[1:])
        if len(extras) > 1:
            raise ValueError("layout constraint accepts one binding set")
    if not dims_node.elts:
        raise ValueError("layout constraint cannot be empty")

    def resolve_extent(item: ast.AST):
        value = _where_static(item, context)
        if isinstance(value, bool) or not isinstance(value, (int, DimVar)):
            raise ValueError("layout dimensions must use `_`, an integer, or a symbolic extent")
        return value

    shape = []
    bindings = []
    for index, item in enumerate(dims_node.elts):
        if isinstance(item, ast.Name) and item.id == "_":
            shape.append(_LAYOUT_WILDCARD)
            continue
        if isinstance(item, ast.Name) and item.id == "D":
            raise ValueError("layout broadcast must use a `{topology @ B()}` binding")
        if isinstance(item, ast.BinOp) and isinstance(item.op, ast.MatMult):
            extent = resolve_extent(item.left)
            topology = _constraint_value(item.right)
            if not isinstance(topology, str) or not topology:
                raise ValueError("layout topology binding must be symbolic")
            shape.append(extent)
            bindings.append((topology, Split(index)))
            continue
        shape.append(resolve_extent(item))
    if extras:
        bindings.extend(_parse_constraint_bindings(extras[0]))
    if len({topology for topology, _ in bindings}) != len(bindings):
        raise ValueError("layout constraint cannot bind one topology more than once")
    return LayoutConstraint(layout=Layout(shape=tuple(shape)), bindings=tuple(bindings))


class WhereAnnotationPattern(ElementPattern):
    element_name = "where_annotation"
    syntax = BranchPattern(
        "where_annotation",
        AstNodePattern(
            ast.Call,
            FieldPattern(
                "func",
                AstNodePattern(
                    ast.Name,
                    FieldPattern("id", LiteralPattern("where")),
                ),
            ),
        ),
        pattern_id="annotation.where",
    )

    @staticmethod
    def construct(match, children, context):
        node = match.node
        source = context.function.source_filename if context.function else "<string>"
        location = SourceLocation(
            filename=source,
            line=getattr(node, "lineno", 0),
            column=getattr(node, "col_offset", 0),
            end_line=getattr(node, "end_lineno", None),
            end_column=getattr(node, "end_col_offset", None),
        )
        if node.args:
            raise ParseError.from_node(node, context, "where(...) accepts keyword arguments only")
        if not node.keywords:
            raise ParseError.from_node(node, context, "where(...) cannot be empty")
        constraints = []
        fields = set()
        try:
            for keyword in node.keywords:
                if keyword.arg is None:
                    raise ValueError("where(...) does not accept **kwargs")
                if keyword.arg in fields:
                    raise ValueError(f"where(...) repeats keyword {keyword.arg!r}")
                fields.add(keyword.arg)
                if keyword.arg == "layout":
                    constraints.append(
                        dataclasses.replace(
                            _parse_layout_constraint(keyword.value, context),
                            source_loc=location,
                            provenance=ConstraintProvenance.AUTHOR,
                        )
                    )
                elif keyword.arg == "mesh":
                    constraints.append(
                        MeshConstraint(
                            mesh=_where_static(keyword.value, context),
                            source_loc=location,
                            provenance=ConstraintProvenance.AUTHOR,
                        )
                    )
                elif keyword.arg == "storage":
                    constraints.append(
                        StorageConstraint(
                            storage=_where_static(keyword.value, context),
                            source_loc=location,
                            provenance=ConstraintProvenance.AUTHOR,
                        )
                    )
                else:
                    raise ValueError(
                        f"where(...) has unknown field {keyword.arg!r}; use "
                        "layout=..., mesh=..., or storage=..."
                    )
        except (TypeError, ValueError) as error:
            raise ParseError.from_node(node, context, str(error)) from error
        return ScheduleConstraintMetadata(constraints=tuple(constraints), source_loc=location)

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class ReturnTypePattern(ElementPattern):
    element_name = "return_type"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "return_type",
            ChildPattern(
                "type",
                lambda: TypeAnnotationPattern(),
                "type_annotation",
                "return",
                values={"position": "hir_output"},
            ),
            pattern_id="function.return_type",
        )
    )

    @staticmethod
    def construct(match, children, context):
        value = children["type"]
        context.lexical_scope.define(_RETURN_TYPE, value)
        return value

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class SignaturePattern(ElementPattern):
    element_name = "signature"
    syntax = LazyPattern(
        lambda: BindPattern(
            AstNodePattern(
                ast.arguments,
                FieldPattern("posonlyargs", SequencePattern()),
                FieldPattern(
                    "args",
                    RepeatPattern(
                        AstNodePattern(
                            ast.arg,
                            FieldPattern(
                                "annotation",
                                AstNodePattern(
                                    ast.expr,
                                    ChildPattern(
                                        "parameter_{index}",
                                        TypeAnnotationPattern(),
                                        "type_annotation",
                                        "parameter",
                                    ),
                                ),
                            ),
                        )
                    ),
                ),
                FieldPattern("vararg", LiteralPattern(None)),
                FieldPattern("kwonlyargs", SequencePattern()),
                FieldPattern("kw_defaults", SequencePattern()),
                FieldPattern("kwarg", LiteralPattern(None)),
                FieldPattern("defaults", SequencePattern()),
            ),
            SignaturePattern._bind,
        )
    )

    @staticmethod
    def _bind(node: object, context: MatchContext, matched: AstMatch[Any]) -> AstMatch[Any] | None:
        assert isinstance(node, ast.arguments)
        if context.function is None:
            return None
        names = tuple(argument.arg for argument in node.args)
        constness = tuple(
            isinstance(argument.annotation, ast.Subscript)
            and TensorPattern._head(argument.annotation.value) == "ConstTensor"
            for argument in node.args
        )
        children: list[AstChild] = []
        for index, argument in enumerate(node.args):
            assert argument.annotation is not None
            if context.function.dialect == "hir":
                position = "hir_input"
            else:
                first_output = max(0, len(node.args) - context.function.output_count)
                position = "tir_output" if index >= first_output else "tir_input"
            children.append(
                AstChild(
                    f"parameter_{index}",
                    TypeAnnotationPattern(),
                    argument.annotation,
                    "type_annotation",
                    "parameter",
                    values={"position": position},
                )
            )
        return dataclasses.replace(
            matched,
            pattern_id="function.signature",
            branch_id="signature",
            captures={
                **matched.captures,
                "names": names,
                "constness": constness,
            },
            children=tuple(children),
        )

    @staticmethod
    def construct(match, children, context):
        names = match.captures["names"]
        constness = match.captures["constness"]
        params = tuple(
            runtime.Var(
                type=children[f"parameter_{index}"],
                name=name,
                is_const=constness[index],
            )
            for index, name in enumerate(names)
        )
        for param in params:
            context.lexical_scope.define(param.name, param)
        return params

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class StaticLiteralPattern(ElementPattern):
    element_name = "literal"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "static_literal",
            AstNodePattern(
                ast.Constant,
                FieldPattern(
                    "value",
                    CapturePattern("value", lambda value, context: value),
                ),
            ),
            pattern_id="static.literal",
        )
    )

    @staticmethod
    def construct(match, children, context):
        return match.captures["value"]

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class StaticReferencePattern(ElementPattern):
    element_name = "primary"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            BranchPattern(
                "static_name",
                AstNodePattern(
                    ast.Name,
                    FieldPattern(
                        "id",
                        CapturePattern("name", lambda value, context: value),
                    ),
                ),
                pattern_id="static.name",
            ),
            BranchPattern(
                "static_attribute",
                AstNodePattern(
                    ast.Attribute,
                    FieldPattern(
                        "attr",
                        CapturePattern("attribute", lambda value, context: value),
                    ),
                    FieldPattern(
                        "value",
                        ChildPattern(
                            "owner",
                            lambda: StaticValuePattern(),
                            "static_owner",
                            "static_owner",
                        ),
                    ),
                ),
                pattern_id="static.attribute",
            ),
        )
    )

    @staticmethod
    def construct(match, children, context):
        if match.branch_id == "static_name":
            return _resolve_reference(match.node, context)
        elif match.branch_id == "static_attribute":
            owner = children["owner"]
            attribute = match.captures["attribute"]
            try:
                return getattr(owner, attribute)
            except AttributeError as error:
                raise ParseError.from_node(
                    match.node,
                    context,
                    f"{type(owner).__name__} has no attribute {attribute!r}",
                ) from error
        raise RuntimeError(f"no constructor branch for {match.branch_id!r}")

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class StaticSequencePattern(ElementPattern):
    element_name = "sequence"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            StaticSequencePattern._branch(ast.Tuple, tuple, "static.tuple"),
            StaticSequencePattern._branch(ast.List, list, "static.list"),
            StaticSequencePattern._branch(ast.Set, set, "static.set"),
        )
    )

    @staticmethod
    def _branch(node_type: type, constructor: type, pattern_id: str) -> AstPattern[Any]:
        return BranchPattern(
            "static_sequence",
            AstNodePattern(
                node_type,
                CapturePattern("constructor", lambda node, context: constructor),
                FieldPattern(
                    "elts",
                    RepeatPattern(
                        ChildPattern(
                            "item_{index}",
                            StaticValuePattern(),
                            "static_item",
                            "static_item",
                        )
                    ),
                ),
            ),
            pattern_id=pattern_id,
        )

    @staticmethod
    def construct(match, children, context):
        return match.captures["constructor"](children.values())

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class StaticDictPattern(ElementPattern):
    element_name = "dict"
    syntax = LazyPattern(
        lambda: BindPattern(
            AstNodePattern(
                ast.Dict,
                FieldPattern("keys", RepeatPattern(StaticValuePattern())),
                FieldPattern("values", RepeatPattern(StaticValuePattern())),
            ),
            StaticDictPattern._bind,
        )
    )

    @staticmethod
    def _bind(
        node: object, context: MatchContext, matched: AstMatch[Any]
    ) -> AstMatch[Any] | MatchFailure | None:
        assert isinstance(node, ast.Dict)
        if any(key is None for key in node.keys):
            return PatternFailure(
                "static_dict", node, "`**` unpacking is not a static dictionary form"
            )
        children: list[AstChild] = []
        for index, (key, value) in enumerate(zip(node.keys, node.values)):
            assert key is not None
            children.extend(
                (
                    AstChild(
                        f"key_{index}",
                        StaticValuePattern(),
                        key,
                        "static_key",
                        "static_key",
                    ),
                    AstChild(
                        f"value_{index}",
                        StaticValuePattern(),
                        value,
                        "static_value",
                        "static_value",
                    ),
                )
            )
        return dataclasses.replace(
            matched,
            pattern_id="static.dict",
            branch_id="static_dict",
            captures={**matched.captures, "length": len(node.keys)},
            children=tuple(children),
        )

    @staticmethod
    def construct(match, children, context):
        return {
            children[f"key_{index}"]: children[f"value_{index}"]
            for index in range(match.captures["length"])
        }

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class StaticBinaryPattern(ElementPattern):
    element_name = "binary_operation"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "static_binary",
            AstNodePattern(
                ast.BinOp,
                FieldPattern(
                    "op",
                    ChoicePattern(
                        AstNodePattern(ast.Add),
                        AstNodePattern(ast.Sub),
                        AstNodePattern(ast.Mult),
                        AstNodePattern(ast.Div),
                        AstNodePattern(ast.FloorDiv),
                        AstNodePattern(ast.Mod),
                        AstNodePattern(ast.Pow),
                    ),
                ),
                CapturePattern("operator", lambda node, context: _BINARY_OPERATORS[type(node.op)]),
                FieldPattern(
                    "left",
                    ChildPattern("left", StaticValuePattern(), "static_operand"),
                ),
                FieldPattern(
                    "right",
                    ChildPattern("right", StaticValuePattern(), "static_operand"),
                ),
            ),
            pattern_id="static.binary",
        )
    )

    @staticmethod
    def construct(match, children, context):
        try:
            return match.captures["operator"](children["left"], children["right"])
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise ParseError.from_node(match.node, context, str(error)) from error

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class StaticUnaryPattern(ElementPattern):
    element_name = "unary_operation"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "static_unary",
            AstNodePattern(
                ast.UnaryOp,
                FieldPattern(
                    "op",
                    ChoicePattern(
                        AstNodePattern(ast.UAdd),
                        AstNodePattern(ast.USub),
                        AstNodePattern(ast.Not),
                    ),
                ),
                CapturePattern("operator", lambda node, context: _UNARY_OPERATORS[type(node.op)]),
                FieldPattern(
                    "operand",
                    ChildPattern("operand", StaticValuePattern(), "static_operand"),
                ),
            ),
            pattern_id="static.unary",
        )
    )

    @staticmethod
    def construct(match, children, context):
        try:
            return match.captures["operator"](children["operand"])
        except (TypeError, ValueError) as error:
            raise ParseError.from_node(match.node, context, str(error)) from error

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class StaticCallPattern(ElementPattern):
    element_name = "call"
    syntax = LazyPattern(
        lambda: BindPattern(
            AstNodePattern(
                ast.Call,
                FieldPattern("func", StaticValuePattern()),
                FieldPattern("args", RepeatPattern(StaticValuePattern())),
                FieldPattern(
                    "keywords",
                    RepeatPattern(
                        AstNodePattern(
                            ast.keyword,
                            FieldPattern(
                                "arg",
                                PredicatePattern(
                                    "keyword-name",
                                    lambda value, context: isinstance(value, str),
                                ),
                            ),
                            FieldPattern("value", StaticValuePattern()),
                        )
                    ),
                ),
            ),
            StaticCallPattern._bind,
        )
    )

    @staticmethod
    def _bind(
        node: object, context: MatchContext, matched: AstMatch[Any]
    ) -> AstMatch[Any] | MatchFailure | None:
        assert isinstance(node, ast.Call)
        keyword_names = [keyword.arg for keyword in node.keywords]
        if len(set(keyword_names)) != len(keyword_names):
            repeated = sorted(
                {name for name in keyword_names if name and keyword_names.count(name) > 1}
            )
            return PatternFailure(
                "static_call", node, f"keyword given more than once: {', '.join(repeated)}"
            )
        children = [AstChild("callee", StaticValuePattern(), node.func, "static_callee")]
        children.extend(
            AstChild(
                f"arg_{index}",
                StaticValuePattern(),
                argument,
                "static_argument",
            )
            for index, argument in enumerate(node.args)
        )
        children.extend(
            AstChild(
                f"kw_{keyword.arg}",
                StaticValuePattern(),
                keyword.value,
                "static_argument",
                keyword.arg,
            )
            for keyword in node.keywords
            if keyword.arg is not None
        )
        return dataclasses.replace(
            matched,
            pattern_id="static.call",
            branch_id="static_call",
            captures={
                **matched.captures,
                "arg_count": len(node.args),
                "keywords": tuple(keyword_names),
            },
            children=tuple(children),
        )

    @staticmethod
    def construct(match, children, context):
        callee = children["callee"]
        if not callable(callee) and not isinstance(callee, type):
            raise ParseError.from_node(
                match.node,
                context,
                "static calls require a callable target",
            )
        args = tuple(children[f"arg_{index}"] for index in range(match.captures["arg_count"]))
        kwargs = {name: children[f"kw_{name}"] for name in match.captures["keywords"]}
        try:
            return callee(*args, **kwargs)
        except (TypeError, ValueError) as error:
            raise ParseError.from_node(match.node, context, str(error)) from error

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class StaticSlicePattern(ElementPattern):
    element_name = "slice"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "static_slice",
            AstNodePattern(
                ast.Slice,
                FieldPattern(
                    "lower",
                    OptionalPattern(
                        ChildPattern("lower", StaticValuePattern(), "static_slice", "lower")
                    ),
                ),
                FieldPattern(
                    "upper",
                    OptionalPattern(
                        ChildPattern("upper", StaticValuePattern(), "static_slice", "upper")
                    ),
                ),
                FieldPattern(
                    "step",
                    OptionalPattern(
                        ChildPattern("step", StaticValuePattern(), "static_slice", "step")
                    ),
                ),
            ),
            pattern_id="static.slice",
        )
    )

    @staticmethod
    def construct(match, children, context):
        return slice(children.get("lower"), children.get("upper"), children.get("step"))

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class StaticSubscriptPattern(ElementPattern):
    element_name = "subscript"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "static_subscript",
            AstNodePattern(
                ast.Subscript,
                FieldPattern(
                    "value",
                    ChildPattern("owner", StaticValuePattern(), "static_owner"),
                ),
                FieldPattern(
                    "slice",
                    ChildPattern("key", StaticValuePattern(), "static_key"),
                ),
            ),
            pattern_id="static.subscript",
        )
    )

    @staticmethod
    def construct(match, children, context):
        try:
            return children["owner"][children["key"]]
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise ParseError.from_node(match.node, context, str(error)) from error

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class StaticValuePattern(ElementPattern):
    element_name = "expression"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            StaticLiteralPattern(),
            StaticReferencePattern(),
            StaticSequencePattern(),
            StaticDictPattern(),
            StaticBinaryPattern(),
            StaticUnaryPattern(),
            StaticCallPattern(),
            StaticSlicePattern(),
            StaticSubscriptPattern(),
        )
    )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class NamePattern(ElementPattern):
    element_name = "name"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "name",
            AstNodePattern(
                ast.Name,
                FieldPattern("id", CapturePattern("name", lambda value, context: value)),
            ),
            pattern_id="expression.name",
        )
    )

    @staticmethod
    def construct(match, children, context):
        name = match.captures["name"]
        value = context.lexical_scope.lookup(name)
        if value is None:
            value = context.function.closure.get(name)
        if isinstance(value, slice):
            value = value.start
        if isinstance(value, runtime.Expr):
            return value
        if isinstance(value, (bool, int, float)):
            return _constant(value)
        raise ParseError.from_node(match.node, context, f"name {name!r} is not an Expr")

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class ConstantPattern(ElementPattern):
    element_name = "constant"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "constant",
            AstNodePattern(
                ast.Constant,
                FieldPattern("value", LiteralPattern(value_type=(bool, int, float))),
            ),
            pattern_id="expression.constant",
        )
    )

    @staticmethod
    def construct(match, children, context):
        return _constant(match.captures["value"])

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class TupleExpressionPattern(ElementPattern):
    element_name = "tuple_expression"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "expr_tuple",
            AstNodePattern(
                ast.Tuple,
                FieldPattern(
                    "elts",
                    RepeatPattern(
                        ChildPattern(
                            "element_{index}",
                            ExpressionPattern(),
                            "expression",
                            "tuple_item",
                        )
                    ),
                ),
            ),
            pattern_id="expression.tuple",
        )
    )

    @staticmethod
    def construct(match, children, context):
        elements = tuple(children.values())
        return runtime.IrTuple(
            type=runtime.TupleType(fields=tuple(item.type for item in elements)),
            elements=elements,
        )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


@dataclass(frozen=True)
class CallBindingRule:
    STATEMENT: ClassVar[str] = "A call must bind its arguments into a Call tuple."

    def apply(self, value, *, match, context):
        if not isinstance(value, runtime.Call):
            raise ParseError.from_node(match.node, context, "call did not construct Call")
        if not isinstance(value.args, tuple):
            raise ParseError.from_node(match.node, context, "call arguments are not a tuple")
        if context.function is not None and context.function.state.mesh_stack:
            value = replace_metadata(
                value,
                ExecutionDomainMetadata(tuple(context.function.state.mesh_stack)),
            )
        return value


def _types_compatible(actual: object, expected: object) -> bool:
    if actual == expected:
        return True
    if isinstance(actual, runtime.TensorType) and isinstance(expected, runtime.TensorType):
        try:
            actual_shape = tuple(runtime.normalize_dim(dim) for dim in actual.shape)
            expected_shape = tuple(runtime.normalize_dim(dim) for dim in expected.shape)
        except (TypeError, ValueError):
            actual_shape = actual.shape
            expected_shape = expected.shape
        return actual_shape == expected_shape and actual.dtype == expected.dtype
    if isinstance(actual, runtime.TupleType) and isinstance(expected, runtime.TupleType):
        return len(actual.fields) == len(expected.fields) and all(
            _types_compatible(left, right) for left, right in zip(actual.fields, expected.fields)
        )
    return False


@dataclass(frozen=True)
class CallTypeInferenceRule:
    STATEMENT: ClassVar[str] = "A call's result type must be inferred from its binding."

    def apply(self, value, *, match, context):
        if not isinstance(value, runtime.Call):
            return value
        value = attach_authored_metadata(value, match.node, context)
        infer_context = context.lexical_scope.lookup(_TYPE_INFER_CONTEXT)
        if not isinstance(infer_context, runtime.TypeInferContext):
            infer_context = runtime.TypeInferContext()
        computed = runtime.TypeInferVisitor(infer_context).visit(value)
        return dataclasses.replace(value, type=computed)


class CallExpectedTypeRule:
    STATEMENT: ClassVar[str] = "A call's inferred type must satisfy the expected expression type."

    def apply(self, value, *, match, context):
        expected = context.expected_type
        actual = getattr(value, "type", None)
        if expected is not None and not _types_compatible(actual, expected):
            raise ParseError.from_node(
                match.node,
                context,
                f"expression type {actual!r} does not match expected type {expected!r}",
            )
        return value


@dataclass(frozen=True)
class VariadicInputs:
    items: tuple[runtime.Expr, ...]

    def __init__(self, values):
        items = tuple(values)
        if not all(isinstance(value, runtime.Expr) for value in items):
            raise TypeError("variadic inputs must contain Expr values")
        object.__setattr__(self, "items", items)


class VariadicInputsPattern(ElementPattern):
    element_name = "variadic_inputs"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            BranchPattern(
                "list",
                BindPattern(
                    AstNodePattern(
                        ast.List,
                        FieldPattern(
                            "elts",
                            RepeatPattern(
                                ChildPattern(
                                    "input_{index}",
                                    ExpressionPattern(),
                                    "variadic_input",
                                    "input",
                                )
                            ),
                        ),
                    ),
                    VariadicInputsPattern._bind_sequence,
                ),
                pattern_id="call.variadic.list",
            ),
            BranchPattern(
                "tuple",
                BindPattern(
                    AstNodePattern(
                        ast.Tuple,
                        FieldPattern(
                            "elts",
                            RepeatPattern(
                                ChildPattern(
                                    "input_{index}",
                                    ExpressionPattern(),
                                    "variadic_input",
                                    "input",
                                )
                            ),
                        ),
                    ),
                    VariadicInputsPattern._bind_sequence,
                ),
                pattern_id="call.variadic.tuple",
            ),
            BindPattern(
                AstNodePattern(ast.ListComp),
                VariadicInputsPattern._bind_comprehension,
            ),
            BranchPattern(
                "generator_expression",
                AstNodePattern(ast.GeneratorExp),
                pattern_id="call.variadic.generator_expression",
            ),
            BranchPattern(
                "unsupported",
                AstNodePattern(ast.expr),
                pattern_id="call.variadic.unsupported",
            ),
        )
    )

    @staticmethod
    def _bind_sequence(
        node: object, context: MatchContext, matched: AstMatch[Any]
    ) -> AstMatch[Any]:
        assert isinstance(node, (ast.List, ast.Tuple))
        if any(isinstance(element, ast.Starred) for element in node.elts):
            raise ParseError.from_node(
                node,
                context,
                "variadic input sequences do not support starred expansion",
            )
        return matched

    @staticmethod
    def _bind_comprehension(
        node: object, context: MatchContext, matched: AstMatch[Any]
    ) -> AstMatch[Any]:
        assert isinstance(node, ast.ListComp)
        if len(node.generators) != 1:
            raise ParseError.from_node(
                node,
                context,
                "variadic list comprehension supports exactly one generator",
            )
        generator = node.generators[0]
        if not isinstance(generator.target, ast.Name):
            raise ParseError.from_node(
                generator.target,
                context,
                "variadic list comprehension requires a simple Name target",
            )
        if generator.ifs or generator.is_async:
            raise ParseError.from_node(
                generator,
                context,
                "variadic list comprehension does not support filters or async generators",
            )
        iterator = generator.iter
        if not (
            isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Name)
            and iterator.func.id == "range"
            and not iterator.keywords
            and 1 <= len(iterator.args) <= 3
        ):
            raise ParseError.from_node(
                iterator,
                context,
                "variadic list comprehension requires range with 1 to 3 static integer arguments",
            )
        values = []
        for argument in iterator.args:
            try:
                value = parse_node(StaticValuePattern(), argument, context)
            except ParseError as error:
                raise ParseError.from_node(
                    argument,
                    context,
                    "variadic list comprehension range arguments must be static integers",
                ) from error
            if not isinstance(value, int) or isinstance(value, bool):
                raise ParseError.from_node(
                    argument,
                    context,
                    "variadic list comprehension range arguments must be static integers",
                )
            values.append(value)
        children = tuple(
            AstChild(
                f"input_{index}",
                ExpressionPattern(),
                node.elt,
                "variadic_input",
                "input",
                lexical_bindings={generator.target.id: value},
                isolated_scope=True,
            )
            for index, value in enumerate(range(*values))
        )
        return dataclasses.replace(
            matched,
            pattern_id="call.variadic.list_comp",
            branch_id="list_comp",
            children=children,
        )

    @staticmethod
    def construct(match, children, context):
        return VariadicInputs(tuple(children.values()))

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


def _variadic_item_annotation(param: object) -> object | None:
    if getattr(param, "kind", None) != "input":
        return None
    annotation = getattr(param, "annotation", None)
    if get_origin(annotation) is not tuple:
        return None
    args = get_args(annotation)
    if len(args) == 1:
        return args[0]
    if len(args) == 2 and args[1] is Ellipsis:
        return args[0]
    return None


@dataclass(frozen=True)
class CallVariadicInputFormRule:
    STATEMENT: ClassVar[str] = (
        "A variadic call must use one explicit list, tuple, or supported static list comprehension."
    )

    def apply(self, value, *, match, context):
        schema = match.captures.get("schema")
        if not isinstance(schema, runtime.OpSchema):
            return value
        inputs = tuple(param for param in schema.signature if param.kind == "input")
        if len(inputs) != 1 or _variadic_item_annotation(inputs[0]) is None:
            return value
        argument = match.node.args[0]
        if isinstance(argument, ast.GeneratorExp):
            raise ParseError.from_node(
                argument,
                context,
                "variadic inputs do not support generator expressions; use a list comprehension",
            )
        if not isinstance(argument, (ast.List, ast.Tuple, ast.ListComp)):
            raise ParseError.from_node(
                argument,
                context,
                "variadic inputs require an explicit list, tuple, or supported list comprehension",
            )
        return value


class CallPattern(ElementPattern):
    element_name = "op_call"
    syntax = LazyPattern(
        lambda: BindPattern(
            AstNodePattern(
                ast.Call,
                FieldPattern("func", ReferencePattern()),
                FieldPattern("args", RepeatPattern(AstNodePattern(ast.expr))),
                FieldPattern(
                    "keywords",
                    RepeatPattern(
                        AstNodePattern(
                            ast.keyword,
                            FieldPattern(
                                "arg",
                                PredicatePattern(
                                    "keyword-name",
                                    lambda value, context: isinstance(value, str),
                                ),
                            ),
                            FieldPattern("value", AstNodePattern(ast.expr)),
                        )
                    ),
                ),
            ),
            CallPattern._bind,
        )
    )

    @staticmethod
    def _pattern_for_param(param: object, node: ast.AST) -> AstPattern[Any]:
        annotation = param.annotation
        if _variadic_item_annotation(param) is not None:
            return VariadicInputsPattern()
        if annotation is runtime.TensorType and isinstance(node, ast.Subscript):
            return TensorPattern()
        if annotation is runtime.DType:
            return DTypePattern()
        if annotation is runtime.StorageKind or param.name == "storage":
            return StoragePattern()
        if annotation in (runtime.Layout, runtime.ShardLayout, runtime.LayoutBase):
            return LayoutPattern()
        return StaticValuePattern()

    @staticmethod
    def _schema_children(
        node: ast.Call, schema: object, context: MatchContext
    ) -> tuple[AstChild, ...] | MatchFailure | None:
        """Bind a call's arguments to one op schema's inputs and attributes.

        A ``Tuple[T]`` input consumes exactly one explicit sequence and flattens
        its elements into ``Call.args``. Attributes remain keyword-only, so the
        sequence boundary cannot be confused with an attribute position.
        """
        params = tuple(schema.signature)
        inputs = [param for param in params if param.kind == "input"]
        attrs = [param for param in params if param.kind == "attribute"]
        variadic = len(inputs) == 1 and _variadic_item_annotation(inputs[0]) is not None
        positional = list(node.args)
        children: list[AstChild] = []
        bound_attrs: set[str] = set()
        if variadic:
            if len(positional) != 1:
                raise ParseError.from_node(
                    node,
                    context,
                    "variadic operations require exactly one list, tuple, or "
                    "supported list-comprehension input sequence",
                )
            children.append(
                AstChild(
                    "variadic_inputs",
                    CallPattern._pattern_for_param(inputs[0], positional[0]),
                    positional[0],
                    "variadic_inputs",
                    "inputs",
                )
            )
            positional = []
        for index, argument in enumerate(positional):
            if index < len(inputs):
                name = inputs[index].name
                children.append(
                    AstChild(
                        f"input_{index}",
                        ExpressionPattern(),
                        argument,
                        "call_argument",
                        name,
                    )
                )
                continue
            attr_index = index - len(inputs)
            if attr_index >= len(attrs):
                return PatternFailure(
                    "op_call",
                    node,
                    f"{schema.name} takes at most {len(inputs) + len(attrs)} positional "
                    f"arguments, got {len(node.args)}",
                )
            param = attrs[attr_index]
            bound_attrs.add(param.name)
            children.append(
                AstChild(
                    f"attr_{param.name}",
                    CallPattern._pattern_for_param(param, argument),
                    argument,
                    "call_attribute",
                    "allocation" if param.annotation is runtime.TensorType else param.name,
                )
            )
        for keyword in node.keywords:
            if keyword.arg is None:
                return PatternFailure(
                    "op_call", node, "`**` unpacking is not an authored call form"
                )
            if keyword.arg in bound_attrs:
                return PatternFailure(
                    "op_call",
                    node,
                    f"attribute {keyword.arg!r} is already bound by a positional argument",
                )
            param = next((item for item in attrs if item.name == keyword.arg), None)
            if param is None:
                known = ", ".join(item.name for item in attrs) or "none"
                return PatternFailure(
                    "op_call",
                    node,
                    f"{schema.name} has no attribute {keyword.arg!r}; its attributes are: {known}",
                )
            bound_attrs.add(param.name)
            children.append(
                AstChild(
                    f"attr_{param.name}",
                    CallPattern._pattern_for_param(param, keyword.value),
                    keyword.value,
                    "call_attribute",
                    "allocation" if param.annotation is runtime.TensorType else param.name,
                )
            )
        if not variadic and len(positional) < len(inputs):
            return PatternFailure(
                "op_call",
                node,
                f"{schema.name} takes {len(inputs)} inputs, got {len(positional)}",
            )
        return tuple(children)

    @staticmethod
    def _bind(
        node: object, context: MatchContext, matched: AstMatch[Any]
    ) -> AstMatch[Any] | MatchFailure | None:
        assert isinstance(node, ast.Call)
        module_owner = None
        try:
            callee = _resolve_reference(node.func, context)
        except ParseError:
            return None
        if isinstance(callee, runtime.Module):
            module_owner = callee
            if node.keywords:
                return PatternFailure(
                    "op_call", node, "a module call takes positional arguments only"
                )
            try:
                callee = callee.entry_function()
            except ValueError as error:
                return PatternFailure("op_call", node.func, str(error))
        if isinstance(callee, runtime.Function):
            if node.keywords:
                return PatternFailure(
                    "op_call", node, "a function call takes positional arguments only"
                )
            return dataclasses.replace(
                matched,
                pattern_id="call.function",
                branch_id="function_call",
                captures={
                    **matched.captures,
                    "callee": callee,
                    "module_owner": locals().get("module_owner"),
                    "module_binding": (node.func.id if isinstance(node.func, ast.Name) else None),
                },
                children=tuple(
                    AstChild(
                        f"arg_{index}",
                        ExpressionPattern(),
                        argument,
                        "call_argument",
                        "argument",
                    )
                    for index, argument in enumerate(node.args)
                ),
            )
        schema = (
            callee if isinstance(callee, runtime.OpSchema) else getattr(callee, "_op_schema", None)
        )
        if not isinstance(schema, runtime.OpSchema):
            return None
        children = CallPattern._schema_children(node, schema, context)
        if children is None or isinstance(children, MatchFailure):
            return children
        return dataclasses.replace(
            matched,
            pattern_id="call.operation",
            branch_id="operation_call",
            captures={**matched.captures, "schema": schema},
            children=children,
        )

    @staticmethod
    def construct(match, children, context):
        if match.branch_id == "operation_call":
            schema = match.captures["schema"]
            variadic_inputs = children.get("variadic_inputs")
            if isinstance(variadic_inputs, VariadicInputs):
                inputs = variadic_inputs.items
            else:
                inputs = tuple(
                    value for name, value in children.items() if name.startswith("input_")
                )
            attrs = {
                name.removeprefix("attr_"): value
                for name, value in children.items()
                if name.startswith("attr_")
            }
            for name, value in tuple(attrs.items()):
                parameter = next((item for item in schema.signature if item.name == name), None)
                annotation = None if parameter is None else parameter.annotation
                if get_origin(annotation) is Literal and value not in get_args(annotation):
                    choices = ", ".join(repr(choice) for choice in get_args(annotation))
                    raise ParseError.from_node(
                        match.node,
                        context,
                        f"{name} must be one of {choices}, got {value!r}",
                    )
                if (
                    isinstance(annotation, type)
                    and issubclass(annotation, enum.Enum)
                    and isinstance(value, str)
                ):
                    try:
                        attrs[name] = annotation(value)
                    except ValueError as error:
                        raise ParseError.from_node(match.node, context, str(error)) from error
            try:
                operation = schema.builder(**attrs)
            except (TypeError, ValueError) as error:
                raise ParseError.from_node(match.node, context, str(error)) from error
            placeholder_type = context.expected_type
            if placeholder_type is None:
                placeholder_type = runtime.TensorType.scalar(runtime.DType.f32)
            return runtime.Call(type=placeholder_type, target=operation, args=inputs)
        elif match.branch_id == "function_call":
            callee = match.captures["callee"]
            args = tuple(children.values())
            module_owner = match.captures.get("module_owner")
            if module_owner is not None and context.function.module is None:
                raise ParseError.from_node(
                    match.node,
                    context,
                    "a Module call is only valid inside a @module class body",
                )
            placeholder = runtime.Call(
                type=callee.return_type,
                target=callee,
                args=args,
            )
            infer_context = context.lexical_scope.lookup(_TYPE_INFER_CONTEXT)
            if not isinstance(infer_context, runtime.TypeInferContext):
                infer_context = runtime.TypeInferContext()
            instance = runtime.elaborate(
                callee, tuple(arg.type for arg in args), infer_context, placeholder
            )
            return dataclasses.replace(placeholder, target=instance)
        raise RuntimeError(f"no constructor branch for {match.branch_id!r}")

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        CallVariadicInputFormRule(),
        CallBindingRule(),
        CallTypeInferenceRule(),
        CallExpectedTypeRule(),
    )


_EXPR_BINARY_KINDS: Mapping[type[ast.AST], str] = {
    ast.Add: "ADD",
    ast.Sub: "SUB",
    ast.Mult: "MUL",
    ast.Div: "DIV",
    ast.FloorDiv: "FLOOR_DIV",
    ast.Mod: "MOD",
    ast.Eq: "EQ",
    ast.NotEq: "NE",
    ast.Lt: "LT",
    ast.LtE: "LE",
    ast.Gt: "GT",
    ast.GtE: "GE",
    ast.And: "AND",
    ast.Or: "OR",
}


def _expression_operator_pattern(operator_type: type[ast.AST]) -> ChoicePattern:
    return ChoicePattern(
        *(
            AstNodePattern(node_type)
            for node_type in _EXPR_BINARY_KINDS
            if issubclass(node_type, operator_type)
        )
    )


_CALL_RESULT_RULES: tuple[AstRule[Any], ...] = (
    CallBindingRule(),
    CallTypeInferenceRule(),
    CallExpectedTypeRule(),
)


class MatMulExpressionPattern(ElementPattern):
    element_name = "matmul_expression"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "matmul_expression",
            AstNodePattern(
                ast.BinOp,
                FieldPattern("op", AstNodePattern(ast.MatMult)),
                FieldPattern(
                    "left",
                    ChildPattern("left", ExpressionPattern(), "expression"),
                ),
                FieldPattern(
                    "right",
                    ChildPattern("right", ExpressionPattern(), "expression"),
                ),
            ),
            pattern_id="expression.matmul",
        )
    )

    @staticmethod
    def construct(match, children, context):
        args = (children["left"], children["right"])
        placeholder_type = next(
            (arg.type for arg in args if getattr(arg, "type", None) is not None),
            runtime.TensorType.scalar(runtime.DType.f32),
        )
        return runtime.Call(
            type=placeholder_type,
            target=MatMul(),
            args=args,
        )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = _CALL_RESULT_RULES


class BinaryExpressionPattern(ElementPattern):
    element_name = "binary_expression"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            BranchPattern(
                "binary_expression",
                AstNodePattern(
                    ast.BinOp,
                    FieldPattern(
                        "op",
                        _expression_operator_pattern(ast.operator),
                    ),
                    CapturePattern(
                        "kind",
                        lambda node, context: _EXPR_BINARY_KINDS[type(node.op)],
                    ),
                    FieldPattern(
                        "left",
                        ChildPattern("left", ExpressionPattern(), "expression"),
                    ),
                    FieldPattern(
                        "right",
                        ChildPattern("right", ExpressionPattern(), "expression"),
                    ),
                ),
                pattern_id="expression.binary",
            ),
            BranchPattern(
                "binary_expression",
                AstNodePattern(
                    ast.Compare,
                    FieldPattern(
                        "ops",
                        SequencePattern(_expression_operator_pattern(ast.cmpop)),
                    ),
                    CapturePattern(
                        "kind",
                        lambda node, context: _EXPR_BINARY_KINDS[type(node.ops[0])],
                    ),
                    FieldPattern(
                        "left",
                        ChildPattern("left", ExpressionPattern(), "expression"),
                    ),
                    FieldPattern(
                        "comparators",
                        SequencePattern(ChildPattern("right", ExpressionPattern(), "expression")),
                    ),
                ),
                pattern_id="expression.binary",
            ),
            BranchPattern(
                "binary_expression",
                AstNodePattern(
                    ast.BoolOp,
                    FieldPattern(
                        "op",
                        _expression_operator_pattern(ast.boolop),
                    ),
                    CapturePattern(
                        "kind",
                        lambda node, context: _EXPR_BINARY_KINDS[type(node.op)],
                    ),
                    FieldPattern(
                        "values",
                        SequencePattern(
                            ChildPattern("left", ExpressionPattern(), "expression"),
                            ChildPattern("right", ExpressionPattern(), "expression"),
                        ),
                    ),
                ),
                pattern_id="expression.binary",
            ),
        )
    )

    @staticmethod
    def construct(match, children, context):
        args = (children["left"], children["right"])
        placeholder_type = next(
            (arg.type for arg in args if getattr(arg, "type", None) is not None),
            runtime.TensorType.scalar(runtime.DType.f32),
        )
        return runtime.Call(
            type=placeholder_type,
            target=runtime.Binary(kind=runtime.BinaryKind[match.captures["kind"]]),
            args=args,
        )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        CallBindingRule(),
        CallTypeInferenceRule(),
        CallExpectedTypeRule(),
    )


class UnaryExpressionPattern(ElementPattern):
    element_name = "unary_expression"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "unary_expression",
            AstNodePattern(
                ast.UnaryOp,
                FieldPattern(
                    "op",
                    PredicatePattern(
                        "unary-op",
                        lambda op, context: type(op) in {ast.USub, ast.Not},
                    ),
                ),
                CapturePattern(
                    "kind",
                    lambda node, context: {ast.USub: "NEG", ast.Not: "NOT"}[type(node.op)],
                ),
                FieldPattern(
                    "operand",
                    ChildPattern("operand", ExpressionPattern(), "expression"),
                ),
            ),
            pattern_id="expression.unary",
        )
    )

    @staticmethod
    def construct(match, children, context):
        operand = children["operand"]
        target = runtime.Unary(kind=runtime.UnaryKind[match.captures["kind"]])
        return runtime.Call(
            type=operand.type,
            target=target,
            args=(operand,),
        )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        CallBindingRule(),
        CallTypeInferenceRule(),
        CallExpectedTypeRule(),
    )


class SliceEndpointBinaryPattern(ElementPattern):
    element_name = "slice_endpoint_binary"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "dimension_binary",
            AstNodePattern(
                ast.BinOp,
                FieldPattern(
                    "op",
                    PredicatePattern(
                        "dim-op",
                        lambda op, context: (
                            type(op) in {ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod}
                        ),
                    ),
                ),
                CapturePattern("operator", lambda node, context: _BINARY_OPERATORS[type(node.op)]),
                CapturePattern(
                    "dimension_operator",
                    lambda node, context: {
                        ast.Add: runtime.DimAdd,
                        ast.Sub: runtime.DimSub,
                        ast.Mult: runtime.DimMul,
                        ast.FloorDiv: runtime.DimFloorDiv,
                        ast.Mod: runtime.DimMod,
                    }[type(node.op)],
                ),
                FieldPattern(
                    "left",
                    ChildPattern("left", IndexEndpointPattern(), "slice_endpoint"),
                ),
                FieldPattern(
                    "right",
                    ChildPattern("right", IndexEndpointPattern(), "slice_endpoint"),
                ),
            ),
            pattern_id="expression.slice_endpoint.binary",
        )
    )

    @staticmethod
    def construct(match, children, context):
        left = children["left"]
        right = children["right"]
        if (
            isinstance(left, slice)
            and type(match.node.op) in {ast.Add, ast.Sub}
            and isinstance(right, (int, runtime.Expr))
        ):
            offset = right
            if type(match.node.op) is ast.Sub:
                offset = runtime.simplify_dim(runtime.DimMul, (-1, offset))
            try:
                start = runtime.simplify_dim(runtime.DimAdd, (left.start, offset))
                stop = runtime.simplify_dim(runtime.DimAdd, (left.stop, offset))
            except (TypeError, ValueError, ZeroDivisionError) as error:
                raise ParseError.from_node(match.node, context, str(error)) from error
            return slice(start, stop, left.step)
        numeric = all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (left, right)
        )
        try:
            if numeric:
                return match.captures["operator"](left, right)
            return runtime.simplify_dim(match.captures["dimension_operator"], (left, right))
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise ParseError.from_node(match.node, context, str(error)) from error

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class IndexEndpointPattern(ElementPattern):
    element_name = "index_endpoint"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            ConditionPattern(
                "constant",
                lambda node, context: isinstance(node, ast.Constant),
                StaticLiteralPattern(),
            ),
            ConditionPattern(
                "name",
                lambda node, context: isinstance(node, ast.Name),
                StaticReferencePattern(),
            ),
            SliceEndpointBinaryPattern(),
            MeshCoordinatePattern(),
            ExpressionPattern(),
            StaticValuePattern(),
        )
    )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


@dataclass(frozen=True)
class TileWindowSliceBoundRule:
    STATEMENT: ClassVar[str] = "A tile window cannot be used as a slice bound."

    def apply(self, value, *, match, context):
        for authored_name, value_name in (
            ("lower", "start"),
            ("upper", "stop"),
            ("step", "step"),
        ):
            if isinstance(getattr(value, value_name), slice):
                raise ParseError.from_node(
                    getattr(match.node, authored_name),
                    context,
                    "a tile loop variable is already a window and cannot be "
                    "used as a slice bound; use x[:, t, :] or bind "
                    "base = t + 0 before slicing",
                )
        return value


class IndexSlicePattern(ElementPattern):
    element_name = "index_slice"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "static_slice",
            AstNodePattern(
                ast.Slice,
                *(
                    FieldPattern(
                        name,
                        OptionalPattern(
                            ChildPattern(
                                name,
                                IndexEndpointPattern(),
                                "slice_endpoint",
                                name,
                            )
                        ),
                    )
                    for name in ("lower", "upper", "step")
                ),
            ),
            pattern_id="expression.index.slice",
        )
    )

    @staticmethod
    def construct(match, children, context):
        return slice(children.get("lower"), children.get("upper"), children.get("step"))

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (TileWindowSliceBoundRule(),)


class SubscriptIndexPattern(ElementPattern):
    element_name = "subscript_index"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            BranchPattern(
                "tuple_children",
                AstNodePattern(
                    ast.Tuple,
                    FieldPattern(
                        "elts",
                        RepeatPattern(
                            ChoicePattern(
                                ConditionPattern(
                                    "slice",
                                    lambda node, context: isinstance(node, ast.Slice),
                                    ChildPattern(
                                        "index_{index}",
                                        IndexSlicePattern(),
                                        "subscript_index",
                                        "subscript_index",
                                    ),
                                ),
                                ChildPattern(
                                    "index_{index}",
                                    IndexEndpointPattern(),
                                    "subscript_index",
                                    "subscript_index",
                                ),
                            )
                        ),
                    ),
                ),
                pattern_id="expression.index.tuple",
            ),
            BranchPattern(
                "identity",
                ChoicePattern(
                    ConditionPattern(
                        "slice",
                        lambda node, context: isinstance(node, ast.Slice),
                        ChildPattern(
                            "value",
                            IndexSlicePattern(),
                            "subscript_index",
                            "subscript_index",
                        ),
                    ),
                    ChildPattern(
                        "value",
                        IndexEndpointPattern(),
                        "subscript_index",
                        "subscript_index",
                    ),
                ),
                pattern_id="expression.index",
            ),
        )
    )

    @staticmethod
    def construct(match, children, context):
        if match.branch_id == "identity":
            return children["value"]
        elif match.branch_id == "tuple_children":
            return tuple(children.values())
        raise RuntimeError(f"no constructor branch for {match.branch_id!r}")

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class SubscriptExpressionPattern(ElementPattern):
    element_name = "subscript_expression"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "subscript_expression",
            AstNodePattern(
                ast.Subscript,
                FieldPattern("value", ChildPattern("value", ExpressionPattern(), "expression")),
                FieldPattern(
                    "slice",
                    ChildPattern("index", SubscriptIndexPattern(), "subscript_index"),
                ),
            ),
            pattern_id="expression.subscript",
        )
    )

    @staticmethod
    def construct(match, children, context):
        value = children["value"]
        index = children["index"]
        if isinstance(value.type, runtime.TupleType):
            if isinstance(index, bool) or not isinstance(index, int):
                raise ParseError.from_node(
                    match.node, context, "Tuple subscript requires an integer literal"
                )
            normalized = index + len(value.type.fields) if index < 0 else index
            return _infer_call(runtime.TupleGetItem(index=normalized), (value,), context)
        if not isinstance(value.type, runtime.TensorType):
            raise ParseError.from_node(
                match.node, context, "subscript requires TensorType or TupleType"
            )
        indices = index if isinstance(index, tuple) else (index,)
        if len(indices) != len(value.type.shape):
            raise ParseError.from_node(
                match.node,
                context,
                f"tensor subscript rank {len(indices)} != tensor rank {len(value.type.shape)}",
            )
        starts = []
        sizes = []
        strides = []
        collapsed = []
        for axis, (component, extent) in enumerate(zip(indices, value.type.shape)):
            if isinstance(component, slice):
                begin = 0 if component.start is None else component.start
                end = extent if component.stop is None else component.stop
                stride = 1 if component.step is None else component.step
                starts.append(runtime.dim_expr(begin))
                sizes.append(_slice_size(begin, end, stride, context, match.node))
                strides.append(stride)
                continue
            if isinstance(component, bool):
                raise ParseError.from_node(match.node, context, "bool is not a tensor index")
            start = runtime.dim_expr(component)
            starts.append(start)
            sizes.append(1)
            strides.append(1)
            collapsed.append(axis)
        starts_expr = runtime.IrTuple(
            type=runtime.TupleType(fields=tuple(start.type for start in starts)),
            elements=tuple(starts),
        )
        sliced = _infer_call(
            runtime.Slice(sizes=tuple(sizes), strides=tuple(strides)),
            (value, starts_expr),
            context,
        )
        if not collapsed:
            return sliced
        new_shape = tuple(
            extent for axis, extent in enumerate(sliced.type.shape) if axis not in collapsed
        )
        return _infer_call(runtime.Reshape(new_shape=new_shape), (sliced,), context)

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class MeshCoordinatePattern(ElementPattern):
    element_name = "mesh_coordinate"
    syntax = LazyPattern(
        lambda: BindPattern(
            AstNodePattern(
                ast.Attribute,
                FieldPattern("value", AstNodePattern(ast.Name)),
                FieldPattern("attr", LiteralPattern(value_type=str)),
            ),
            MeshCoordinatePattern._bind,
        )
    )

    @staticmethod
    def _bind(
        node: object, context: MatchContext, matched: AstMatch[Any]
    ) -> AstMatch[Any] | MatchFailure | None:
        assert isinstance(node, ast.Attribute)
        assert isinstance(node.value, ast.Name)
        if context.function is None:
            return None
        mesh = context.lexical_scope.lookup(node.value.id)
        if not isinstance(mesh, runtime.Mesh):
            return None
        axis = next(
            (index for index, name in enumerate(mesh.names) if name == node.attr),
            None,
        )
        if axis is None and node.attr in {"x", "y", "z"}:
            candidate = ("x", "y", "z").index(node.attr)
            if candidate < len(mesh.layout.shape):
                axis = candidate
        if axis is None:
            named = ", ".join(mesh.names)
            return PatternFailure(
                "mesh_coordinate",
                node,
                f"mesh {node.value.id!r} has no axis {node.attr!r}; its axes are: {named}",
            )
        return dataclasses.replace(
            matched,
            pattern_id="expression.mesh_coordinate",
            branch_id="mesh_coordinate",
            captures={**matched.captures, "mesh": mesh, "axis": axis},
        )

    @staticmethod
    def construct(match, children, context):
        if context.function is None:
            raise ParseError.from_node(match.node, context, "mesh coordinate lacks context")
        mesh = match.captures["mesh"]
        axis = match.captures["axis"]
        extent = mesh.layout.shape[axis]
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise ParseError.from_node(
                match.node, context, "mesh coordinate requires a concrete axis extent"
            )
        cache_key = (id(mesh), axis)
        cached = context.function.state.mesh_coordinates.get(cache_key)
        if cached is not None:
            return cached
        vector_type = runtime.TensorType(
            shape=(extent,),
            dtype=runtime.DType.i64,
            layout=None,
            storage=runtime.StorageKind.GMEM,
        )
        vector = _infer_call(runtime.Arange(type=vector_type), (), context)
        attrs = tuple(
            runtime.Split(axis=0) if mesh_axis == axis else runtime.Broadcast()
            for mesh_axis in range(len(mesh.layout.shape))
        )
        layout = runtime.ShardLayout(
            layout=runtime.Layout(shape=(extent,), strides=(1,)),
            attrs=attrs,
            mesh=mesh,
        )
        placed = _infer_call(
            runtime.Reshard(layout=layout, storage=runtime.StorageKind.RMEM),
            (vector,),
            context,
        )
        local = _infer_call(runtime.Local(), (placed,), context)
        coordinate = _infer_call(runtime.Reshape(new_shape=()), (local,), context)
        context.function.state.mesh_coordinates[cache_key] = coordinate
        return coordinate

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class ExpressionPattern(ElementPattern):
    element_name = "runtime_expression"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            CallPattern(),
            LaunchPattern(),
            SubscriptExpressionPattern(),
            MatMulExpressionPattern(),
            BinaryExpressionPattern(),
            UnaryExpressionPattern(),
            MeshCoordinatePattern(),
            NamePattern(),
            ConstantPattern(),
            TupleExpressionPattern(),
            TensorPattern(),
            BranchPattern(
                "attribute_expr",
                AstNodePattern(ast.Attribute),
                pattern_id="expression.attribute",
            ),
        )
    )

    @staticmethod
    def construct(match, children, context):
        value = _resolve_reference(match.node, context)
        if isinstance(value, runtime.Expr):
            return value
        if isinstance(value, (bool, int, float)):
            return _constant(value)
        raise ParseError.from_node(match.node, context, "attribute did not resolve to Expr")

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class MeshContextPattern(ElementPattern):
    element_name = "mesh_context"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            BindPattern(
                AstNodePattern(
                    ast.Call,
                    FieldPattern(
                        "func",
                        ChoicePattern(
                            AstNodePattern(
                                ast.Name,
                                FieldPattern("id", LiteralPattern("Mesh")),
                            ),
                            AstNodePattern(
                                ast.Attribute,
                                FieldPattern("attr", LiteralPattern("Mesh")),
                            ),
                        ),
                    ),
                    FieldPattern("args", RepeatPattern(AstNodePattern(ast.expr), minimum=1)),
                    FieldPattern(
                        "keywords",
                        RepeatPattern(
                            AstNodePattern(
                                ast.keyword,
                                FieldPattern(
                                    "arg",
                                    ChoicePattern(
                                        LiteralPattern("layout"),
                                        LiteralPattern("names"),
                                    ),
                                ),
                            )
                        ),
                    ),
                ),
                MeshContextPattern._bind,
            ),
            BranchPattern(
                "mesh_reference",
                AstNodePattern(
                    ast.expr,
                    PredicatePattern(
                        "reference",
                        lambda node, context: isinstance(
                            node, (ast.Name, ast.Attribute, ast.Subscript)
                        ),
                    ),
                    ChildPattern(
                        "value",
                        StaticValuePattern(),
                        "static_mesh",
                        "mesh",
                    ),
                ),
                pattern_id="mesh.reference",
            ),
        )
    )

    @staticmethod
    def _bind(
        node: object, context: MatchContext, matched: AstMatch[Any]
    ) -> AstMatch[Any] | MatchFailure | None:
        assert isinstance(node, ast.Call)
        if not node.args or not isinstance(node.args[0], ast.Tuple):
            return None
        positional = list(node.args[1:])
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        layout_node = keywords.get("layout")
        names_node = keywords.get("names")
        if positional:
            if layout_node is not None:
                return PatternFailure(
                    "mesh", node, "`layout` is already bound by a positional argument"
                )
            layout_node = positional.pop(0)
        if positional:
            if names_node is not None:
                return PatternFailure(
                    "mesh", node, "`names` is already bound by a positional argument"
                )
            names_node = positional.pop(0)
        if positional:
            return PatternFailure(
                "mesh",
                node,
                f"a mesh takes topologies, layout, and names; got "
                f"{len(node.args)} positional arguments",
            )
        if layout_node is None:
            return PatternFailure("mesh", node, "a mesh requires a `layout`")
        children = [
            AstChild(
                "topology_names",
                StaticValuePattern(),
                node.args[0],
                "mesh_topologies",
                "topologies",
            ),
            AstChild("layout", LayoutPattern(), layout_node, "mesh_layout", "layout"),
        ]
        if names_node is not None:
            children.append(
                AstChild(
                    "names",
                    StaticValuePattern(),
                    names_node,
                    "mesh_names",
                    "names",
                )
            )
        return dataclasses.replace(
            matched,
            pattern_id="mesh.context",
            branch_id="mesh_context",
            children=tuple(children),
        )

    @staticmethod
    def construct(match, children, context):
        if match.branch_id == "mesh_context":
            if context.function is None:
                raise ParseError.from_node(match.node, context, "Mesh requires function context")
            topology_names = children["topology_names"]
            if not isinstance(topology_names, tuple):
                raise ParseError.from_node(
                    match.node,
                    context,
                    "Mesh topologies must be a tuple",
                )
            if all(isinstance(name, str) for name in topology_names):
                try:
                    topologies = tuple(context.function.topologies[name] for name in topology_names)
                except KeyError as error:
                    raise ParseError.from_node(
                        match.node,
                        context,
                        f"topology {error.args[0]!r} not declared by @module",
                    ) from error
            elif all(hasattr(topology, "name") for topology in topology_names):
                topologies = topology_names
            else:
                raise ParseError.from_node(
                    match.node,
                    context,
                    "Mesh topologies must be names or Topology objects",
                )
            names = children.get("names", ())
            try:
                mesh = runtime.Mesh(
                    topologies=topologies,
                    layout=children["layout"],
                    names=names,
                )
            except (TypeError, ValueError) as error:
                raise ParseError.from_node(match.node, context, str(error)) from error
            binding = context.values.get("mesh_binding")
            if isinstance(binding, str):
                context.lexical_scope.define(binding, mesh)
            context.function.state.mesh_stack.append(mesh)
            return mesh
        elif match.branch_id == "mesh_reference":
            if context.function is None:
                raise ParseError.from_node(match.node, context, "Mesh requires function context")
            mesh = children["value"]
            if not isinstance(mesh, runtime.Mesh):
                raise ParseError.from_node(match.node, context, "with context is not Mesh")
            binding = context.values.get("mesh_binding")
            if isinstance(binding, str):
                context.lexical_scope.define(binding, mesh)
            context.function.state.mesh_stack.append(mesh)
            return mesh
        raise RuntimeError(f"no constructor branch for {match.branch_id!r}")

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class WithPattern(ElementPattern):
    element_name = "with"
    syntax = LazyPattern(
        lambda: BindPattern(
            AstNodePattern(
                ast.With,
                FieldPattern(
                    "items",
                    SequencePattern(
                        AstNodePattern(
                            ast.withitem,
                            FieldPattern("optional_vars", AstNodePattern(ast.Name)),
                            FieldPattern(
                                "context_expr",
                                ChildPattern(
                                    "mesh",
                                    MeshContextPattern(),
                                    "mesh_context",
                                    "mesh",
                                ),
                            ),
                        )
                    ),
                ),
                FieldPattern(
                    "body",
                    ChildPattern(
                        "body",
                        BlockPattern(),
                        "block",
                        "with_body",
                        transform=_body_as_ast_module,
                    ),
                ),
            ),
            WithPattern._bind,
        )
    )

    @staticmethod
    def _bind(node: object, context: MatchContext, matched: AstMatch[Any]) -> AstMatch[Any] | None:
        assert isinstance(node, ast.With)
        item = node.items[0]
        assert isinstance(item.optional_vars, ast.Name)
        binding = item.optional_vars.id
        return dataclasses.replace(
            matched,
            pattern_id="statement.with_mesh",
            branch_id="with_mesh",
            captures={**matched.captures, "binding": binding},
            children=(
                AstChild(
                    "mesh",
                    MeshContextPattern(),
                    item.context_expr,
                    "mesh_context",
                    "mesh",
                    values={"mesh_binding": binding},
                ),
                AstChild(
                    "body",
                    BlockPattern(),
                    _body_as_ast_module(node.body),
                    "block",
                    "with_body",
                ),
            ),
        )

    @staticmethod
    def construct(match, children, context):
        if context.function is None or not context.function.state.mesh_stack:
            raise ParseError.from_node(match.node, context, "Mesh stack is unbalanced")
        mesh = context.function.state.mesh_stack.pop()
        if context.function.dialect == "hir":
            return children["body"]
        binding = runtime.Var(
            type=runtime.TensorType.scalar(runtime.DType.i64, storage=runtime.StorageKind.RMEM),
            name=match.captures["binding"],
        )
        return runtime.MeshScope(mesh=mesh, binding=binding, body=children["body"])

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class LaunchPattern(ElementPattern):
    """TIR host launch statement lowered from the authored ``launch`` call."""

    element_name = "launch"
    syntax = LazyPattern(lambda: BindPattern(AstNodePattern(ast.Call), LaunchPattern._bind))

    @staticmethod
    def _bind(node: object, context: MatchContext, matched: AstMatch[Any]):
        assert isinstance(node, ast.Call)
        if not isinstance(node.func, ast.Name) or node.func.id != "launch":
            return None
        if not node.args:
            return PatternFailure("launch", node, "launch(...) requires a callee")
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        if any(name is None for name in keywords):
            return PatternFailure("launch", node, "`**` unpacking is not an authored launch form")
        missing = [name for name in ("grid", "block") if name not in keywords]
        if missing:
            return PatternFailure("launch", node, f"launch(...) requires {' and '.join(missing)}")
        unknown = set(keywords) - {
            "grid",
            "block",
            "cluster",
            "dynamic_smem",
            "stream",
            "attrs",
        }
        if unknown:
            return PatternFailure(
                "launch",
                node,
                f"launch(...) has no option {sorted(unknown)!r}; it takes grid, block, "
                f"cluster, dynamic_smem, stream, attrs",
            )
        children = [
            AstChild("callee", StaticValuePattern(), node.args[0], "launch_callee"),
            *(
                AstChild(f"arg_{index}", ExpressionPattern(), argument, "launch_argument")
                for index, argument in enumerate(node.args[1:])
            ),
            AstChild("grid", StaticValuePattern(), keywords["grid"], "launch_extent"),
            AstChild("block", StaticValuePattern(), keywords["block"], "launch_extent"),
        ]
        for name in ("cluster", "dynamic_smem", "stream", "attrs"):
            if name in keywords:
                children.append(
                    AstChild(name, StaticValuePattern(), keywords[name], "launch_option")
                )
        return dataclasses.replace(
            matched,
            pattern_id="statement.launch",
            branch_id="launch",
            captures={"arg_count": len(node.args) - 1},
            children=tuple(children),
        )

    @staticmethod
    def construct(match, children, context):
        callee = children["callee"]
        if isinstance(callee, runtime.Module):
            callee = callee.entry_function()
        options = {
            name: children[name]
            for name in ("cluster", "dynamic_smem", "stream", "attrs")
            if name in children
        }
        return launch_call(
            callee,
            tuple(children[f"arg_{index}"] for index in range(match.captures["arg_count"])),
            children["grid"],
            children["block"],
            **options,
        )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class LoopCarryStatementPattern(ElementPattern):
    element_name = "loop_carry_statement"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            BranchPattern(
                "loop_carry_statement",
                AstNodePattern(
                    ast.Assign,
                    FieldPattern(
                        "targets",
                        SequencePattern(AstNodePattern(ast.expr)),
                    ),
                    CapturePattern("names", LoopCarryStatementPattern._target_names),
                ),
                pattern_id="loop.carry_statement",
            ),
            BranchPattern(
                "loop_carry_statement",
                AstNodePattern(
                    ast.For,
                    CapturePattern("names", lambda node, context: ()),
                    FieldPattern(
                        "body",
                        ChildPattern(
                            "nested",
                            LoopCarryPattern(),
                            "loop_carry",
                            transform=_body_as_ast_module,
                        ),
                    ),
                ),
                pattern_id="loop.carry_statement",
            ),
            BranchPattern(
                "loop_carry_statement",
                AstNodePattern(
                    ast.stmt,
                    CapturePattern("names", lambda node, context: ()),
                ),
                pattern_id="loop.carry_statement",
            ),
        )
    )

    @staticmethod
    def _target_names(node: object, context: MatchContext) -> tuple[str, ...]:
        assert isinstance(node, ast.Assign)
        target = node.targets[0]
        targets = target.elts if isinstance(target, ast.Tuple) else (target,)
        return tuple(item.id for item in targets if isinstance(item, ast.Name))

    @staticmethod
    def construct(match, children, context):
        return (*match.captures["names"], *children.get("nested", ()))

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class LoopCarryPattern(ElementPattern):
    element_name = "loop_carry"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "loop_carry",
            AstNodePattern(
                ast.Module,
                FieldPattern(
                    "body",
                    RepeatPattern(
                        ChildPattern(
                            "statement_{index}",
                            LoopCarryStatementPattern(),
                            "loop_carry_statement",
                        )
                    ),
                ),
            ),
            pattern_id="loop.carry",
        )
    )

    @staticmethod
    def construct(match, children, context):
        names: list[str] = []
        for statement_names in children.values():
            for name in statement_names:
                if name not in names:
                    names.append(name)
        return tuple(names)

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class LoopIteratorPattern(ElementPattern):
    element_name = "loop_iterator"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            BranchPattern(
                "tile",
                AstNodePattern(
                    ast.Call,
                    FieldPattern(
                        "func",
                        AstNodePattern(ast.Name, FieldPattern("id", LiteralPattern("tile"))),
                    ),
                    FieldPattern("keywords", SequencePattern()),
                    FieldPattern(
                        "args",
                        SequencePattern(AstNodePattern(ast.expr), AstNodePattern(ast.expr)),
                    ),
                ),
                pattern_id="loop.iterator.tile",
            ),
            BranchPattern(
                "range",
                AstNodePattern(
                    ast.Call,
                    FieldPattern(
                        "func",
                        AstNodePattern(ast.Name, FieldPattern("id", LiteralPattern("range"))),
                    ),
                    FieldPattern("keywords", SequencePattern()),
                    FieldPattern(
                        "args",
                        ChoicePattern(
                            SequencePattern(AstNodePattern(ast.expr)),
                            SequencePattern(AstNodePattern(ast.expr), AstNodePattern(ast.expr)),
                            SequencePattern(
                                AstNodePattern(ast.expr),
                                AstNodePattern(ast.expr),
                                AstNodePattern(ast.expr),
                            ),
                        ),
                    ),
                ),
                pattern_id="loop.iterator.range",
            ),
        )
    )

    @staticmethod
    def construct(match, children, context):
        return match.branch_id

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class LoopHeaderPattern(ElementPattern):
    element_name = "loop_header"
    syntax = LazyPattern(
        lambda: BindPattern(
            AstNodePattern(
                ast.For,
                FieldPattern("target", AstNodePattern(ast.Name)),
                FieldPattern("iter", LoopIteratorPattern()),
                FieldPattern(
                    "body",
                    ChildPattern(
                        "carry",
                        LoopCarryPattern(),
                        "loop_carry",
                        transform=_body_as_ast_module,
                    ),
                ),
            ),
            LoopHeaderPattern._bind,
        )
    )

    def match(self, node: object, context: MatchContext) -> AstMatch[Any] | MatchFailure | None:
        """Name the invalid iterator before the shape-exact syntax rejects it.

        The syntax encodes each iterator's arity so the generated grammar shows
        what the parser accepts. A shape mismatch alone would report only that
        the pattern did not match, so the specific reason is stated here first.
        """
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
        ):
            kind = node.iter.func.id
            count = len(node.iter.args)
            if kind not in {"tile", "range"}:
                return PatternFailure(
                    "loop_header",
                    node.iter.func,
                    "loop iterator must be tile(...) or range(...)",
                )
            if node.iter.keywords:
                return PatternFailure(
                    "loop_header",
                    node.iter,
                    "tile()/range() does not accept keyword args (positional-only at the IR level)",
                )
            if (kind == "tile" and count != 2) or (kind == "range" and count not in {1, 2, 3}):
                if kind == "tile" and count == 1:
                    detail = "tile(extent) is not supported; use range(extent)"
                elif kind == "tile":
                    detail = f"tile() takes 2 arguments (extent, step), got {count}"
                else:
                    detail = f"range() takes 1 to 3 arguments, got {count}"
                return PatternFailure("loop_header", node.iter, detail)
        return super().match(node, context)

    @staticmethod
    def _bind(
        node: object, context: MatchContext, matched: AstMatch[Any]
    ) -> AstMatch[Any] | MatchFailure | None:
        assert isinstance(node, ast.For)
        assert isinstance(node.target, ast.Name)
        assert isinstance(node.iter, ast.Call)
        assert isinstance(node.iter.func, ast.Name)
        kind = node.iter.func.id
        count = len(node.iter.args)
        if node.iter.keywords:
            return PatternFailure(
                "loop_header",
                node.iter,
                "tile()/range() does not accept keyword args (positional-only at the IR level)",
            )
        if (kind == "tile" and count != 2) or (kind == "range" and count not in {1, 2, 3}):
            if kind == "tile" and count == 1:
                detail = "tile(extent) is not supported; use range(extent)"
            elif kind == "tile":
                detail = f"tile() takes 2 arguments (extent, step), got {count}"
            else:
                detail = f"range() takes 1 to 3 arguments, got {count}"
            return PatternFailure(
                "loop_header",
                node.iter,
                detail,
            )
        if kind == "tile":
            fields = ("extent", "step")
            defaults = {"start": 0}
        elif count == 1:
            fields = ("extent",)
            defaults = {"start": 0, "step": 1}
        elif count == 2:
            fields = ("start", "extent")
            defaults = {"step": 1}
        else:
            fields = ("start", "extent", "step")
            defaults = {}
        children = [
            AstChild(
                "carry",
                LoopCarryPattern(),
                _body_as_ast_module(node.body),
                "loop_carry",
            )
        ]
        children.extend(
            AstChild(
                field_name,
                StaticValuePattern(),
                argument,
                "loop_bound",
                field_name,
            )
            for field_name, argument in zip(fields, node.iter.args)
        )
        return dataclasses.replace(
            matched,
            pattern_id=f"loop.header.{kind}",
            branch_id="loop_header",
            captures={
                **matched.captures,
                "kind": kind,
                "target": node.target.id,
                "fields": fields,
                "defaults": defaults,
            },
            children=tuple(children),
        )

    @staticmethod
    def construct(match, children, context):
        if context.function is None or context.function.dialect != "hir":
            raise ParseError.from_node(match.node, context, "loops require HIR context")
        values = dict(match.captures["defaults"])
        values.update((name, value) for name, value in children.items() if name != "carry")
        try:
            start = runtime.normalize_dim(values["start"])
            extent = runtime.normalize_dim(values["extent"])
            step = runtime.normalize_dim(values["step"])
        except (TypeError, ValueError) as error:
            raise ParseError.from_node(match.node, context, str(error)) from error
        induction_var = runtime.Var(
            type=runtime.TensorType.scalar(runtime.DType.i64),
            name=match.captures["target"],
        )
        carry_names = tuple(
            name
            for name in children["carry"]
            if isinstance(context.lexical_scope.lookup(name), runtime.Expr)
        )
        init_args = tuple(context.lexical_scope.lookup(name) for name in carry_names)
        phi_vars = tuple(
            runtime.Var(type=value.type, name=name) for name, value in zip(carry_names, init_args)
        )
        context.lexical_scope.push_frame()
        if match.captures["kind"] == "tile":
            stop = runtime.simplify_dim(runtime.DimAdd, (induction_var, runtime.dim_expr(step)))
            binding = slice(induction_var, stop, 1)
        else:
            binding = induction_var
        context.lexical_scope.define(match.captures["target"], binding)
        for name, phi in zip(carry_names, phi_vars):
            context.lexical_scope.define(name, phi)
        return LoopFrame(
            kind=match.captures["kind"],
            target=match.captures["target"],
            induction_var=induction_var,
            start=start,
            extent=extent,
            step=step,
            carry_names=carry_names,
            phi_vars=phi_vars,
            init_args=init_args,
        )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class LoopBodyPattern(ElementPattern):
    element_name = "loop_body"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "loop_body",
            AstNodePattern(
                ast.Module,
                PredicatePattern(
                    "assignment-suite",
                    lambda node, context: (
                        not any(
                            isinstance(statement, (ast.Return, ast.With, ast.Expr, ast.Pass))
                            for statement in node.body
                        )
                    ),
                ),
                FieldPattern(
                    "body",
                    RepeatPattern(
                        ChildPattern(
                            "statement_{index}",
                            StatementPattern(),
                            "loop_statement",
                            "loop_statement",
                        )
                    ),
                ),
            ),
            pattern_id="loop.body",
        )
    )

    @staticmethod
    def construct(match, children, context):
        if not children:
            raise ParseError.from_node(match.node, context, "loop body cannot be empty")
        value = tuple(children.values())[-1]
        if not isinstance(value, runtime.Expr):
            raise ParseError.from_node(match.node, context, "loop body must yield an Expr")
        return value

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class ForPattern(ElementPattern):
    element_name = "for"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "loop",
            AstNodePattern(
                ast.For,
                ChildPattern("header", LoopHeaderPattern(), "loop_header"),
                FieldPattern(
                    "body",
                    ChildPattern(
                        "body",
                        LoopBodyPattern(),
                        "loop_body",
                        transform=_body_as_ast_module,
                    ),
                ),
            ),
            pattern_id="statement.for",
        )
    )

    @staticmethod
    def construct(match, children, context):
        frame = children["header"]
        body = children["body"]
        yield_values = tuple(context.lexical_scope.lookup(name) for name in frame.carry_names)
        context.lexical_scope.pop_frame()
        if frame.carry_names:
            result_type = (
                frame.phi_vars[0].type
                if len(frame.phi_vars) == 1
                else runtime.TupleType(fields=tuple(phi.type for phi in frame.phi_vars))
            )
        else:
            result_type = body.type
        grid = runtime.GridRegionExpr(
            type=result_type,
            induction_var=frame.induction_var,
            carried_args=frame.phi_vars,
            init_args=frame.init_args,
            body=body,
            yield_values=yield_values,
            start=frame.start,
            extent=frame.extent,
            step=frame.step,
        )
        if len(frame.carry_names) == 1:
            context.lexical_scope.define(frame.carry_names[0], grid)
        else:
            for index, name in enumerate(frame.carry_names):
                projection = _infer_call(runtime.TupleGetItem(index=index), (grid,), context)
                context.lexical_scope.define(name, projection)
        return grid

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class TupleAssignmentPattern(ElementPattern):
    element_name = "tuple_assignment"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "tuple_assignment",
            AstNodePattern(
                ast.Assign,
                FieldPattern(
                    "targets",
                    SequencePattern(
                        AstNodePattern(
                            ast.Tuple,
                            FieldPattern(
                                "elts",
                                RepeatPattern(AstNodePattern(ast.Name), minimum=1),
                            ),
                        )
                    ),
                ),
                CapturePattern(
                    "names",
                    lambda node, context: tuple(item.id for item in node.targets[0].elts),
                ),
                FieldPattern(
                    "value",
                    ChildPattern(
                        "value",
                        ExpressionPattern(),
                        "expression",
                        "assignment_value",
                    ),
                ),
            ),
            pattern_id="statement.tuple_assign",
        )
    )

    @staticmethod
    def construct(match, children, context):
        value = children["value"]
        names = match.captures["names"]
        if not isinstance(value.type, runtime.TupleType):
            raise ParseError.from_node(
                match.node, context, "tuple assignment requires a TupleType value"
            )
        if len(names) != len(value.type.fields):
            raise ParseError.from_node(
                match.node, context, "tuple assignment arity does not match TupleType"
            )
        schema = (
            getattr(type(value.target), "_op_schema", None)
            if isinstance(value, runtime.Call)
            else None
        )
        parent_name = getattr(schema, "name", None) or ", ".join(names)
        value = replace_metadata(value, BindingMetadata(parent_name))
        for index, name in enumerate(names):
            projection = _infer_call(runtime.TupleGetItem(index=index), (value,), context)
            projection = replace_metadata(projection, BindingMetadata(name))
            context.lexical_scope.define(name, projection)
        return value

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class StatementPattern(ElementPattern):
    element_name = "statement"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            ForPattern(),
            WithPattern(),
            TupleAssignmentPattern(),
            BranchPattern(
                "assignment",
                AstNodePattern(
                    ast.Assign,
                    FieldPattern(
                        "targets",
                        SequencePattern(
                            AstNodePattern(
                                ast.Name,
                                FieldPattern(
                                    "id",
                                    CapturePattern("name", lambda value, context: value),
                                ),
                            )
                        ),
                    ),
                    FieldPattern(
                        "value",
                        ChildPattern(
                            "value",
                            ChoicePattern(ExpressionPattern(), StaticValuePattern()),
                            "expression",
                            "assignment_value",
                        ),
                    ),
                ),
                pattern_id="statement.assign",
            ),
            BranchPattern(
                "assignment",
                AstNodePattern(
                    ast.AnnAssign,
                    FieldPattern(
                        "target",
                        AstNodePattern(
                            ast.Name,
                            FieldPattern(
                                "id",
                                CapturePattern("name", lambda value, context: value),
                            ),
                        ),
                    ),
                    FieldPattern(
                        "annotation",
                        ChildPattern(
                            "annotation",
                            ChoicePattern(WhereAnnotationPattern(), TypeAnnotationPattern()),
                            "annotation",
                            "local",
                        ),
                    ),
                    FieldPattern(
                        "value",
                        OptionalPattern(
                            ChildPattern(
                                "value",
                                ChoicePattern(ExpressionPattern(), StaticValuePattern()),
                                "expression",
                                "assignment_value",
                            )
                        ),
                    ),
                ),
                pattern_id="statement.annassign",
            ),
            BranchPattern(
                "return",
                AstNodePattern(
                    ast.Return,
                    FieldPattern(
                        "value",
                        OptionalPattern(
                            ChildPattern(
                                "value",
                                ExpressionPattern(),
                                "expression",
                                "return_value",
                            )
                        ),
                    ),
                ),
                pattern_id="statement.return",
            ),
            BranchPattern(
                "expr_statement",
                AstNodePattern(
                    ast.Expr,
                    FieldPattern(
                        "value",
                        ChildPattern(
                            "value",
                            ExpressionPattern(),
                            "expression",
                            "statement_value",
                        ),
                    ),
                ),
                pattern_id="statement.expr",
            ),
            BranchPattern("none", AstNodePattern(ast.Pass), pattern_id="statement.pass"),
        )
    )

    @staticmethod
    def construct(match, children, context):
        if match.branch_id == "assignment":
            value = children.get("value")
            annotation = children.get("annotation")
            if value is None:
                raise ParseError.from_node(match.node, context, "assignment requires a value")
            if context.function.dialect == "tir" and not isinstance(value, runtime.Expr):
                context.lexical_scope.define(match.captures["name"], value)
                return None
            if isinstance(annotation, ScheduleConstraintMetadata):
                if context.function.dialect != "hir" or not isinstance(value.type, TensorType):
                    raise ParseError.from_node(
                        match.node,
                        context,
                        "where annotation requires a tensor-valued HIR Expr",
                    )
                previous = get_metadata(value, ScheduleConstraintMetadata)
                if previous is not None:
                    binding = get_metadata(value, BindingMetadata)
                    label = binding.name if binding is not None else "<unnamed>"
                    raise ParseError.from_node(
                        match.node,
                        context,
                        f"duplicate where annotation for Expr {label!r}",
                    )
                value = replace_metadata(value, BindingMetadata(match.captures["name"]))
                object.__setattr__(value, "metadata", (*value.metadata, annotation))
            elif annotation is not None and value.type != annotation:
                raise ParseError.from_node(
                    match.node, context, "annotated assignment type mismatch"
                )
            name = match.captures["name"]
            if context.function.dialect == "hir":
                if isinstance(value, runtime.Call) and get_metadata(value, BindingMetadata) is None:
                    value = replace_metadata(value, BindingMetadata(name))
                context.lexical_scope.define(name, value)
                return value
            variable = runtime.Var(type=value.type, name=name)
            context.lexical_scope.define(name, variable)
            return runtime.LetStmt(variable, value, runtime.Sequential(body=()))
        elif match.branch_id == "return":
            if context.function.dialect == "tir":
                if "value" in children:
                    raise ParseError.from_node(match.node, context, "prim_func return must be bare")
                return runtime.Return()
            if "value" not in children:
                raise ParseError.from_node(match.node, context, "func return must carry a value")
            return children["value"]
        elif match.branch_id == "expr_statement":
            value = children["value"]
            if context.function.dialect == "hir":
                raise ParseError.from_node(
                    match.node, context, "HIR does not allow expression statements"
                )
            if isinstance(value, runtime.Evaluate):
                return value
            if not isinstance(value, runtime.Call) or not isinstance(value.type, runtime.UnitType):
                raise ParseError.from_node(
                    match.node, context, "TIR expression statement must be unit Call"
                )
            return runtime.Evaluate(callable=value.target, args=value.args)
        elif match.branch_id == "none":
            return None
        raise RuntimeError(f"no constructor branch for {match.branch_id!r}")

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class BlockPattern(ElementPattern):
    element_name = "block"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "block",
            AstNodePattern(
                ast.Module,
                CapturePattern(
                    "pass_only",
                    lambda node, context: (
                        len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
                    ),
                ),
                CapturePattern(
                    "terminal_children",
                    lambda node, context: tuple(
                        f"statement_{index}"
                        for index, statement in enumerate(node.body)
                        if isinstance(statement, (ast.Return, ast.With, ast.For))
                    ),
                ),
                FieldPattern(
                    "body",
                    RepeatPattern(
                        ChildPattern(
                            "statement_{index}",
                            StatementPattern(),
                            "statement",
                            "statement",
                        )
                    ),
                ),
            ),
            pattern_id="function.block",
        )
    )

    @staticmethod
    def construct(match, children, context):
        values = list(children.values())
        if context.function.dialect == "hir":
            if match.captures["pass_only"]:
                return None
            for child_name in reversed(match.captures["terminal_children"]):
                value = children[child_name]
                if value is not None:
                    return value
            if context.role == "with_body":
                return None
            raise ParseError.from_node(match.node, context, "HIR body must end with return")

        def fold(index: int) -> list[object]:
            output: list[object] = []
            while index < len(values):
                value = values[index]
                if value is None:
                    index += 1
                    continue
                if isinstance(value, runtime.LetStmt):
                    nested = runtime.Sequential(body=tuple(fold(index + 1)))
                    output.append(dataclasses.replace(value, body=nested))
                    return output
                output.append(value)
                index += 1
            return output

        return runtime.Sequential(body=tuple(fold(0)))

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


@dataclass(frozen=True)
class FunctionSignatureRule:
    STATEMENT: ClassVar[str] = "A function must construct an ordered parameter tuple."

    def apply(self, value, *, match, context):
        if not isinstance(getattr(value, "params", None), tuple):
            raise ParseError.from_node(match.node, context, "function params were not constructed")
        return value


@dataclass(frozen=True)
class FunctionReturnRule:
    STATEMENT: ClassVar[str] = "A HIR function body's inferred type must match its return type."

    def apply(self, value, *, match, context):
        if isinstance(value, runtime.Function) and value.body is not None:
            infer_context = context.lexical_scope.lookup(_TYPE_INFER_CONTEXT)
            if not isinstance(infer_context, runtime.TypeInferContext):
                infer_context = runtime.TypeInferContext()
            body_type = runtime.TypeInferVisitor(infer_context).visit(value.body)
            if not _types_compatible(body_type, value.return_type):
                raise ParseError.from_node(
                    match.node,
                    context,
                    f"function body type {body_type!r} does not match return {value.return_type!r}",
                )
        return value


@dataclass(frozen=True)
class FunctionDialectRule:
    STATEMENT: ClassVar[str] = (
        "A function kind and constructed value must agree with the active dialect."
    )

    def apply(self, value, *, match, context):
        kind = context.function.function_kind
        if context.function.dialect == "hir" and kind == "prim_func":
            raise ParseError.from_node(match.node, context, "prim_func requires tir dialect")
        if context.function.dialect == "tir" and kind != "prim_func":
            raise ParseError.from_node(match.node, context, f"{kind} requires hir dialect")
        expected = runtime.Function if context.function.dialect == "hir" else runtime.PrimFunction
        if not isinstance(value, expected):
            raise ParseError.from_node(
                match.node,
                context,
                f"{context.function.dialect} context constructed {type(value).__name__}",
            )
        return value


@dataclass(frozen=True)
class FunctionRoleValidationRule:
    STATEMENT: ClassVar[str] = (
        "A root, variant, or converter must satisfy its role before registration."
    )

    @staticmethod
    def _validate_standalone(
        function: object,
        function_context: FuncParserContext,
        node: ast.AST,
        match_context: MatchContext,
    ) -> None:
        if function_context.role is FunctionRole.ROOT:
            return
        base = function_context.base
        if not isinstance(base, runtime.Function):
            raise ParseError.from_node(node, match_context, "standalone role lacks a HIR base")
        if getattr(base, "_sealed", False):
            raise ParseError.from_node(node, match_context, f"base {base.name!r} is sealed")
        if getattr(function, "body", None) is None:
            raise ParseError.from_node(
                node,
                match_context,
                f"{function_context.role.value} must have a real body",
            )
        if function_context.role is FunctionRole.CONVERTER and not isinstance(
            function_context.key, str
        ):
            raise ParseError.from_node(node, match_context, "converter key must be a weight name")

    def apply(self, value, *, match, context):
        if context.function is None:
            raise ParseError.from_node(match.node, context, "function lacks parser context")
        if context.function.module is not None:
            context.function.module.validate_function(value, context.function)
        else:
            self._validate_standalone(value, context.function, match.node, context)
        return value


@dataclass(frozen=True)
class FunctionRegistrationRule:
    STATEMENT: ClassVar[str] = (
        "A validated function must be registered exactly once in its owning scope."
    )

    @staticmethod
    def _commit_standalone(
        function: object,
        function_context: FuncParserContext,
    ) -> None:
        if function_context.role is FunctionRole.VARIANT:
            assert function_context.base is not None
            function_context.base.add_variant(function)
        elif function_context.role is FunctionRole.CONVERTER:
            assert function_context.base is not None
            function_context.base.add_converter(function_context.key, function)

    def apply(self, value, *, match, context):
        if context.function is None:
            raise ParseError.from_node(match.node, context, "function lacks parser context")
        if context.function.module is not None:
            context.function.module.commit_function(value, context.function)
        else:
            self._commit_standalone(value, context.function)
        return value


class FunctionPattern(ElementPattern):
    element_name = "function"
    syntax = LazyPattern(
        lambda: BindPattern(
            AstNodePattern(
                ast.FunctionDef,
                FieldPattern("name", CapturePattern("name", lambda value, context: value)),
                FieldPattern(
                    "args",
                    ChildPattern("signature", SignaturePattern(), "signature", "function"),
                ),
                FieldPattern(
                    "returns",
                    OptionalPattern(
                        ChildPattern(
                            "return",
                            ReturnTypePattern(),
                            "type_annotation",
                            "return",
                        )
                    ),
                ),
                FieldPattern(
                    "body",
                    ChildPattern(
                        "body",
                        BlockPattern(),
                        "block",
                        "body",
                        transform=lambda body: _body_as_ast_module(body, strip_docstring=True),
                    ),
                ),
            ),
            FunctionPattern._bind,
        )
    )

    @staticmethod
    def _bind(node: object, context: MatchContext, matched: AstMatch[Any]) -> AstMatch[Any] | None:
        assert isinstance(node, ast.FunctionDef)
        function_context = context.function
        if function_context is None:
            return None
        active_context = context.child(
            situation="function",
            role=function_context.function_kind,
            function=function_context,
        )
        return dataclasses.replace(
            matched,
            pattern_id="function",
            branch_id="function",
            construct_context=active_context,
        )

    @staticmethod
    def construct(match, children, context):
        if context.function is None:
            raise ParseError.from_node(match.node, context, "function lacks context")
        params = children["signature"]
        body = children["body"]
        declared_return = children.get("return")
        specializations = context.function.specializations
        converter = context.function.converter
        if context.function.dialect == "hir":
            if declared_return is None:
                if body is None:
                    raise ParseError.from_node(
                        match.node,
                        context,
                        "HIR pass prototype requires a return annotation",
                    )
                declared_return = body.type
            elif (
                isinstance(declared_return, runtime.TensorType)
                and isinstance(body, runtime.Expr)
                and isinstance(body.type, runtime.TensorType)
                and declared_return.shape == body.type.shape
                and declared_return.dtype == body.type.dtype
                and declared_return.storage == body.type.storage
                and isinstance(declared_return.layout, runtime.ShardLayout)
                and isinstance(body.type.layout, runtime.ShardLayout)
                and isinstance(declared_return.layout.layout, runtime.Layout)
                and isinstance(body.type.layout.layout, runtime.Layout)
                and declared_return.layout.attrs == body.type.layout.attrs
                and declared_return.layout.mesh == body.type.layout.mesh
                and declared_return.layout.layout.strides is None
                and body.type.layout.layout.strides is not None
            ):
                declared_return = runtime.TensorType(
                    shape=declared_return.shape,
                    dtype=declared_return.dtype,
                    layout=body.type.layout,
                    storage=declared_return.storage,
                )
            function_name = (
                getattr(context.function.base, "name", None)
                or context.function.base_name
                or match.captures["name"]
            )
            function = runtime.Function.build(
                name=function_name,
                params=params,
                body=body,
                return_type=declared_return,
                specializations=specializations,
            )
            if context.function.role is FunctionRole.VARIANT:
                object.__setattr__(function, runtime.DISPLAY_NAME, match.captures["name"])
                object.__setattr__(function, "name", function_name)
            elif context.function.role is FunctionRole.CONVERTER:
                object.__setattr__(function, "name", f"{function_name}.converter[{converter}]")
            binding = context.function.binding_name or match.captures["name"]
            define = getattr(context.function.module_scope, "define", None)
            if callable(define):
                define(binding, function)
            return function
        if declared_return is not None:
            raise ParseError.from_node(match.node, context, "prim_func cannot return a value type")
        kwargs = {}
        if context.function.target is not None:
            kwargs["target"] = context.function.target
        function = runtime.PrimFunction(
            name=match.captures["name"],
            params=params,
            body=body,
            output_count=context.function.output_count,
            **kwargs,
        )
        define = getattr(context.function.module_scope, "define", None)
        if callable(define):
            define(context.function.binding_name or match.captures["name"], function)
        return function

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        FunctionSignatureRule(),
        FunctionReturnRule(),
        FunctionDialectRule(),
        FunctionRoleValidationRule(),
        FunctionRegistrationRule(),
    )


def _body_as_ast_module(body: object, *, strip_docstring: bool = False) -> ast.Module:
    """Wrap a statement list as an AST Module with explicit docstring policy."""
    assert isinstance(body, list)
    if (
        strip_docstring
        and body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return ast.Module(body=body, type_ignores=[])


__all__ = [
    "BinaryExpressionPattern",
    "BlockPattern",
    "CallBindingRule",
    "CallExpectedTypeRule",
    "CallPattern",
    "CallTypeInferenceRule",
    "ConstantPattern",
    "DTypePattern",
    "DimExprPattern",
    "ExplicitLayoutPattern",
    "ExpressionPattern",
    "ForPattern",
    "FunctionDialectRule",
    "FunctionPattern",
    "FunctionRegistrationRule",
    "FunctionReturnRule",
    "FunctionRoleValidationRule",
    "FunctionSignatureRule",
    "IndexEndpointPattern",
    "IndexSlicePattern",
    "LayoutPattern",
    "LoopBodyPattern",
    "LoopCarryPattern",
    "LoopCarryStatementPattern",
    "LoopHeaderPattern",
    "MeshAxisPattern",
    "MeshContextPattern",
    "MeshCoordinatePattern",
    "NamePattern",
    "PlacedLayoutPattern",
    "PlainLayoutPattern",
    "ReturnTypePattern",
    "ScalarTypePattern",
    "ShapePattern",
    "SignaturePattern",
    "StatementPattern",
    "StaticBinaryPattern",
    "StaticCallPattern",
    "StaticDictPattern",
    "StaticLiteralPattern",
    "StaticReferencePattern",
    "StaticSequencePattern",
    "StaticSlicePattern",
    "StaticSubscriptPattern",
    "StaticUnaryPattern",
    "StaticValuePattern",
    "StoragePattern",
    "SubscriptExpressionPattern",
    "SubscriptIndexPattern",
    "TensorOptionalSlotPattern",
    "TensorPattern",
    "TensorShapeLayoutPattern",
    "TupleAssignmentPattern",
    "TupleExpressionPattern",
    "TypeAnnotationPattern",
    "UnaryExpressionPattern",
    "CallVariadicInputFormRule",
    "VariadicInputs",
    "VariadicInputsPattern",
    "WithPattern",
]
