"""Concrete executable AST Pattern nodes.

Pattern matching and construction live together here; shared parser state,
rules, and runtime helpers remain in ast_pattern.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Literal, get_args, get_origin

from tilefoundry.ir import clause, core, dim, expr, pattern, tir, types, utils
from tilefoundry.ir.hir import nn
from tilefoundry.ir.types import layout_algebra

from .ast_pattern import (
    BINARY_OPERATORS,
    TYPE_INFER_CONTEXT,
    UNARY_OPERATORS,
    AstChild,
    AstMatch,
    AstNodePattern,
    AstPattern,
    AstRule,
    AuthoredInferContext,
    BindPattern,
    BranchPattern,
    CanonicalDTypeRule,
    CapturePattern,
    ChildPattern,
    ChoicePattern,
    ConditionPattern,
    DeclarationOrderError,
    ElementPattern,
    FieldPattern,
    FuncContext,
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
    attach_authored_metadata,
    authored_source_range,
    constant,
    infer_call,
    parse_node,
    resolve_mesh_topologies_at,
    resolve_reference,
    runtime,
    slice_size,
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
                            lambda node, context: BINARY_OPERATORS[type(node.op)],
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

    def construct(self, match, children, context):
        if match.branch_id == "dim_literal":
            return match.captures["value"]
        if match.branch_id == "dim_name":
            return resolve_reference(match.node, context)
        if match.branch_id == "dim_reference":
            return resolve_reference(match.node, context)
        if match.branch_id == "dim_binary":
            try:
                left, right = children["left"], children["right"]
                if isinstance(left, runtime.Expr) or isinstance(right, runtime.Expr):
                    dimension_op = {
                        ast.Add: "add",
                        ast.Sub: "sub",
                        ast.Mult: "mul",
                        ast.FloorDiv: "floordiv",
                        ast.Mod: "mod",
                    }.get(type(match.node.op))
                    if dimension_op is not None:
                        return runtime.simplify_dim(dimension_op, (left, right))
                return match.captures["operator"](left, right)
            except (TypeError, ValueError, ZeroDivisionError) as error:
                raise ParseError.from_node(match.node, context, str(error)) from error
        if match.branch_id == "dim_call":
            callee = resolve_reference(match.node.func, context)
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

    def construct(self, match, children, context):
        if match.branch_id == "shape_reference":
            return resolve_reference(match.node, context)
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

    def construct(self, match, children, context):
        if match.branch_id == "dtype_literal":
            try:
                return runtime.DType.from_name(match.node.value)
            except ValueError as error:
                raise ParseError.from_node(match.node, context, str(error)) from error
        elif match.branch_id == "dtype_reference":
            if isinstance(match.node, ast.Name) and match.node.id in runtime.dtype_members():
                return runtime.DType.from_name(match.node.id)
            value = resolve_reference(match.node, context)
            if isinstance(value, str):
                try:
                    return runtime.DType.from_name(value)
                except ValueError as error:
                    raise ParseError.from_node(match.node, context, str(error)) from error
            return value
        raise RuntimeError(f"no constructor branch for {match.branch_id!r}")

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (CanonicalDTypeRule(),)


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

    def construct(self, match, children, context):
        shape = tuple(children.values())
        if context.situation == "mesh_layout":
            return runtime.Layout(shape, types.compact_row_major(shape))
        return runtime.Layout(shape)

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        LayoutShapeRule(),
        LayoutPositionRule(),
    )


@dataclass(frozen=True)
class MeshAxisBindingRule:
    STATEMENT: ClassVar[str] = "A placement axis resolves to a lexical Mesh binding."

    def apply(self, value, *, match, context):
        mesh, binding, _ = value
        if not isinstance(mesh, runtime.Mesh):
            raise ParseError.from_node(
                match.node, context, f"{binding!r} is not a lexical Mesh binding"
            )
        return value


@dataclass(frozen=True)
class BareMeshAxisRule:
    STATEMENT: ClassVar[str] = "A bare Mesh placement names a one-axis mesh."

    def apply(self, value, *, match, context):
        mesh, _, axis_name = value
        if axis_name is None and len(types.flatten(mesh.layout).shape) != 1:
            raise ParseError.from_node(
                match.node, context, "bare Mesh placement requires a one-axis mesh"
            )
        return value


@dataclass(frozen=True)
class MeshAxisNameRule:
    STATEMENT: ClassVar[str] = "A named placement axis exists in its lexical Mesh."

    def apply(self, value, *, match, context):
        mesh, binding, axis_name = value
        if axis_name is None:
            return mesh, 0
        try:
            return mesh, mesh.names.index(axis_name)
        except ValueError as error:
            raise ParseError.from_node(
                match.node, context, f"Mesh {binding!r} has no axis {axis_name!r}"
            ) from error


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

    def construct(self, match, children, context):
        node = match.node
        binding = node.id if isinstance(node, ast.Name) else node.value.id
        axis_name = None if isinstance(node, ast.Name) else node.attr
        return context.lexical_scope.lookup_mesh(binding), binding, axis_name

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        MeshAxisBindingRule(), BareMeshAxisRule(), MeshAxisNameRule(),
    )


def meshes_outermost_first(meshes, context):
    """Order placement meshes according to the module's topology declaration."""
    declared = tuple(context.function.topologies) if context.function is not None else ()
    order = {name: index for index, name in enumerate(declared)}

    def rank(mesh):
        levels = tuple(getattr(topology, "name", topology) for topology in mesh.topologies)
        return min((order.get(level, len(order)) for level in levels), default=len(order))

    return tuple(sorted(meshes, key=rank))


@dataclasses.dataclass(frozen=True)
class PlacedLayout:
    """A placement written as sugar: the shape the author wrote, and where it goes.

    The two are separate answers and neither reproduces the other. ``shape`` is
    the extent tuple as written, before any mesh axis divides it, which is what
    a Tensor type states. ``layout`` is the placement that sugar implies, whose
    own shape is the divided one. Reading a logical shape back off the layout
    gives the divided shape instead, which is a different type.
    """

    shape: tuple
    layout: object


def layout_dims() -> AstPattern[Any]:
    """``(extent, extent @ axis, ...)`` — the layout's own divided dimensions."""
    built = AstNodePattern(
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
                                        "elts", RepeatPattern(MeshAxisPattern(), minimum=1)
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
        grammar_name="layout_dims",
    )
    return built


def layout_strides() -> AstPattern[Any]:
    """``(stride, ...)`` — how the divided positions are addressed."""
    return AstNodePattern(
        ast.Tuple,
        FieldPattern("elts", RepeatPattern(DimExprPattern())),
        grammar_name="layout_strides",
    )


def state_call(name: str, *args) -> AstPattern[Any]:
    return AstNodePattern(
        ast.Call,
        FieldPattern("func", AstNodePattern(ast.Name, FieldPattern("id", LiteralPattern(name)))),
        FieldPattern("args", SequencePattern(*args)),
        FieldPattern("keywords", SequencePattern()),
    )


def value_states() -> AstPattern[Any]:
    """One Broadcast call, or axis-bound Partial/Varying calls."""
    return ChoicePattern(
        AstNodePattern(ast.Set, FieldPattern("elts", SequencePattern(state_call("B")))),
        AstNodePattern(
            ast.Set,
            FieldPattern(
                "elts",
                RepeatPattern(
                    AstNodePattern(
                        ast.BinOp,
                        FieldPattern("op", AstNodePattern(ast.MatMult)),
                        FieldPattern("left", MeshAxisPattern()),
                        FieldPattern(
                            "right",
                            ChoicePattern(
                                state_call(
                                    "P",
                                    AstNodePattern(
                                        ast.Constant,
                                        FieldPattern("value", LiteralPattern(value_type=str)),
                                    ),
                                ),
                                state_call("V"),
                            ),
                        ),
                    ),
                    minimum=1,
                ),
            ),
        ),
    )


@dataclass(frozen=True)
class ValueStatesFormRule:
    STATEMENT: ClassVar[str] = (
        "Layout states are either a single B() or one or more axis @ P(nonempty string) "
        "or axis @ V() "
        "entries; empty states, axis @ B(), and mixing B() with other entries are rejected."
    )

    def apply(self, value, *, node, context):
        if isinstance(node, ast.Set) and len(node.elts) == 1:
            item = node.elts[0]
            if isinstance(item, ast.Call) and isinstance(item.func, ast.Name):
                if item.func.id == "B" and not item.args and not item.keywords:
                    return value
        if (
            isinstance(node, ast.Set)
            and node.elts
            and all(value_state_parts(item) is not None for item in node.elts)
        ):
            return value
        raise ParseError.from_node(
            node, context, 'layout states must be {B()} or {axis @ P("reduction") | axis @ V(), ...}'
        )


class ValueStatesPattern(ElementPattern):
    element_name = "value_states"
    syntax = LazyPattern(value_states)
    BIND_RULES = (ValueStatesFormRule(),)
    RULES = ()

    def match(self, node, context):
        for rule in self.BIND_RULES:
            rule.apply(None, node=node, context=context)
        return super().match(node, context)


def layout_sugar_parts(node: object):
    """Read a dimension tuple, optionally paired with an explicit stride tuple."""
    if not isinstance(node, ast.Tuple):
        return None
    if len(node.elts) == 2 and all(isinstance(item, ast.Tuple) for item in node.elts):
        return node.elts[0], node.elts[1]
    return node, None


def value_state_parts(node: ast.AST):
    """Read ``axis @ P("reduction")`` or ``axis @ V()`` as axis node and state."""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.MatMult):
        return None
    call = node.right
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.keywords:
        return None
    if call.func.id == "V" and not call.args:
        return node.left, "V", None
    if call.func.id == "P" and len(call.args) == 1:
        reduction = call.args[0]
        if isinstance(reduction, ast.Constant) and isinstance(reduction.value, str):
            if reduction.value:
                return node.left, "P", reduction.value
    return None


@dataclass(frozen=True)
class PlacementCandidate:
    shape: tuple
    strides: tuple | None
    splits: tuple[tuple[object, int, int], ...]
    states: tuple[tuple[object, int, str, str | None], ...]
    states_written: bool = False
    mesh: types.Mesh | None = None
    attrs: tuple[object, ...] = ()
    composed: types.ComposedLayout | None = None


def placement_meshes(value: PlacementCandidate, context: MatchContext, match):
    referenced_ids = {id(entry[0]) for entry in (*value.splits, *value.states)}
    if not referenced_ids:
        return ()
    if context.function is None:
        raise ParseError.from_node(match.node, context, "placed layout requires function context")
    meshes = tuple(dict.fromkeys(entry[0] for entry in (*value.splits, *value.states)))
    if len(meshes) != len(referenced_ids):
        raise ParseError.from_node(
            match.node, context, "placement references a non-lexical Mesh binding"
        )
    return meshes


@dataclass(frozen=True)
class LayoutStrideRankRule:
    STATEMENT: ClassVar[str] = (
        "A stated stride tuple must have the rank of the layout it addresses."
    )

    def apply(self, value, *, match, context):
        if value.strides is not None and len(value.shape) != len(value.strides):
            raise ParseError.from_node(match.node, context, "layout shape/stride rank mismatch")
        return value


@dataclass(frozen=True)
class PlacementMeshResolutionRule:
    STATEMENT: ClassVar[str] = "A placement's mesh must be a lexical mesh binding."

    def apply(self, value, *, match, context):
        placement_meshes(value, context, match)
        return value


@dataclass(frozen=True)
class PlacementLevelRule:
    STATEMENT: ClassVar[str] = "A placement's meshes cannot name the same topology level."

    def apply(self, value, *, match, context):
        meshes = placement_meshes(value, context, match)
        levels = [topology.name for mesh in meshes for topology in mesh.topologies]
        if len(levels) != len(set(levels)):
            duplicates = sorted(name for name in set(levels) if levels.count(name) > 1)
            raise ParseError.from_node(
                match.node,
                context,
                f"a layout can split one level once; two of these meshes name {duplicates}",
            )
        return value


@dataclass(frozen=True)
class MeshAxisBoundOnceRule:
    STATEMENT: ClassVar[str] = "A placement binds each mesh axis at most once."

    def apply(self, value, *, match, context):
        seen: set[tuple[int, int]] = set()
        for mesh, axis, *_ in (*value.splits, *value.states):
            key = (id(mesh), axis)
            if key in seen:
                raise ParseError.from_node(match.node, context, "mesh axis is bound more than once")
            seen.add(key)
        return value


def placement_scope(context):
    if context.function is None or context.situation == "mesh_layout":
        return None
    return authored_infer_context(context).current_mesh


@dataclass(frozen=True)
class PlacementScopeRule:
    STATEMENT: ClassVar[str] = (
        "A placement requires and covers the current mesh scope, or the function "
        "mesh in a signature; each referenced axis must "
        "map uniquely to an axis of that scope. Unstated axes are Broadcast."
    )

    def apply(self, value, *, match, context):
        if not value.splits and not value.states and not value.states_written:
            return value
        mesh = placement_scope(context)
        if mesh is None:
            raise ParseError.from_node(
                match.node, context, "placement requires a current mesh scope"
            )
        keys = types.axis_keys(mesh)
        attrs = [runtime.Broadcast() for _ in keys]
        for source, source_axis, *state in (*value.splits, *value.states):
            key = types.axis_keys(source)[source_axis]
            targets = [axis for axis, held in enumerate(keys) if held == key]
            if len(targets) != 1:
                raise ParseError.from_node(
                    match.node,
                    context,
                    "placement axis does not map uniquely to the current mesh scope",
                )
            if len(state) == 1:
                attr = runtime.Split(state[0])
            else:
                kind, reduction = state
                attr = types.Varying() if kind == "V" else runtime.Partial(reduction)
            attrs[targets[0]] = attr
        return dataclasses.replace(value, mesh=mesh, attrs=tuple(attrs))


@dataclass(frozen=True)
class PlacementConstructionRule:
    """Materialize the candidate only after placement invariants have run."""

    STATEMENT: ClassVar[str] = (
        "A placement must construct a valid shard layout. Without splits, partials "
        "or stated strides, it uses row-major strides; Broadcast states do not "
        "affect this default. A composed operand is retained directly without "
        "canonical split decomposition."
    )

    def apply(self, value, *, match, context):
        if value.composed is not None:
            return PlacedLayout(
                shape=value.shape,
                layout=runtime.ShardLayout(value.composed, value.attrs, value.mesh),
            )
        if value.mesh is None:
            return PlacedLayout(
                shape=value.shape, layout=runtime.Layout(shape=value.shape, strides=value.strides)
            )
        if value.strides is not None:
            return PlacedLayout(
                value.shape,
                runtime.ShardLayout(
                    runtime.Layout(value.shape, value.strides), value.attrs, value.mesh
                ),
            )
        try:
            storage = context.values.get("layout_storage", runtime.StorageKind.GMEM)
            target = None if context.function is None else context.function.target
            layout = runtime.ShardLayout.from_logical(
                value.shape,
                value.mesh,
                value.attrs,
                storage=storage,
                owners=target.storage_owners if target else expr.StorageOwners((), {}),
            )
        except (TypeError, ValueError) as error:
            raise ParseError.from_node(match.node, context, str(error)) from error
        return PlacedLayout(value.shape, layout)


@dataclass(frozen=True)
class PlacementAnswerRule:
    """A placement answers both halves: the shape written, and where it goes."""

    STATEMENT: ClassVar[str] = (
        "Placement sugar states both the shape as written and the layout it implies."
    )

    def apply(self, value, *, match, context):
        if not isinstance(value, PlacedLayout):
            raise ParseError.from_node(match.node, context, "placement did not answer a placement")
        if not isinstance(value.shape, tuple):
            raise ParseError.from_node(match.node, context, "placement shape is not a tuple")
        if not isinstance(value.layout, runtime.LayoutBase):
            raise ParseError.from_node(match.node, context, "placement did not carry a LayoutBase")
        return value


def placement_operand_syntax():
    return ChoicePattern(
        AstNodePattern(
            ast.Tuple,
            FieldPattern("elts", SequencePattern(layout_dims(), layout_strides())),
        ),
        layout_dims(),
    )


def placement_parts(node: ast.AST) -> tuple[ast.AST, tuple[ast.AST, ...]] | None:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.MatMult):
        return None
    left = placement_parts(node.left)
    if left is None:
        extent = node.left
        axes: tuple[ast.AST, ...] = ()
    else:
        extent, axes = left
    right_axes = tuple(node.right.elts) if isinstance(node.right, ast.Tuple) else (node.right,)
    return extent, (*axes, *right_axes)


def enter_placed_layout(
    node: object,
    context: MatchContext,
    matched: AstMatch[Any],
    *,
    allow_plain: bool = False,
) -> AstMatch[Any] | None:
    parts = layout_sugar_parts(node)
    if parts is None:
        return None
    dims_node, strides_node = parts
    children: list[AstChild] = []
    bindings: list[tuple[str, int]] = []
    found_placement = False
    for tensor_axis, item in enumerate(dims_node.elts):
        placement = placement_parts(item)
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
                AstChild(child_name, MeshAxisPattern(), axis_node, "mesh_axis", "mesh_axis")
            )
    if strides_node is not None:
        for index, item in enumerate(strides_node.elts):
            stride_context = context.child(situation="layout_strides", role="layout_strides")
            if DimExprPattern().match(item, stride_context) is None:
                return None
            children.append(
                AstChild(
                    f"stride_{index}",
                    DimExprPattern(),
                    item,
                    "layout_strides",
                    "layout_strides",
                )
            )
    if not allow_plain and not found_placement and strides_node is None:
        return None
    return dataclasses.replace(
        matched,
        pattern_id="tensor.layout.placed",
        branch_id="placed_layout",
        captures={
            **matched.captures,
            "rank": len(dims_node.elts),
            "stride_rank": None if strides_node is None else len(strides_node.elts),
            "bindings": tuple(bindings),
        },
        children=tuple(children),
    )


class PlacedLayoutPattern(ElementPattern):
    """The one placement production, including dimensions, strides, and states."""

    element_name = "placed_layout"
    syntax = LazyPattern(
        lambda: BindPattern(
            placement_operand_syntax(),
            enter_placed_layout,
        )
    )

    def construct(self, match, children, context):
        rank = match.captures["rank"]
        shape = tuple(children[f"extent_{axis}"] for axis in range(rank))
        stride_rank = match.captures.get("stride_rank")
        strides = (
            None
            if stride_rank is None
            else tuple(children[f"stride_{index}"] for index in range(stride_rank))
        )
        splits = tuple(
            (*children[child_name], tensor_axis)
            for child_name, tensor_axis in match.captures["bindings"]
        )
        return PlacementCandidate(shape, strides, splits, ())

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        LayoutStrideRankRule(),
        MeshAxisBoundOnceRule(),
        PlacementMeshResolutionRule(),
        PlacementLevelRule(),
        PlacementScopeRule(),
        PlacementConstructionRule(),
        PlacementAnswerRule(),
    )


def bind_placement_operand(node, context, matched):
    return enter_placed_layout(node, context, matched, allow_plain=True)


class PlacementOperandPattern(PlacedLayoutPattern):
    """Parse placement dimensions before the enclosing state rules construct them."""

    element_name = "placement_operand"
    syntax = LazyPattern(
        lambda: BindPattern(
            placement_operand_syntax(),
            bind_placement_operand,
        )
    )
    RULES = ()

@dataclass(frozen=True)
class StatedLayoutOperandRule:
    STATEMENT: ClassVar[str] = (
        "States attach to placement dimensions, a plain Layout, or a ComposedLayout "
        "whose outer is a plain Layout."
    )

    def apply(self, value, *, match, context):
        operand, states = value
        if isinstance(operand, PlacementCandidate):
            return dataclasses.replace(operand, states=states, states_written=True)
        if isinstance(operand, types.ComposedLayout):
            return PlacementCandidate(
                operand.shape,
                None,
                (),
                states,
                True,
                composed=operand,
            )
        if not isinstance(operand, types.Layout):
            raise ParseError.from_node(
                match.node, context, "layout states require a Layout or placement operand"
            )
        return PlacementCandidate(
            operand.shape,
            operand.strides,
            (),
            states,
            True,
        )


def bind_stated_layout(node, context, matched):
    children = list(matched.children)
    states = []
    for index, item in enumerate(node.right.elts):
        state = value_state_parts(item)
        if state is None:
            continue
        axis, kind, reduction = state
        name = f"state_{index}"
        children.append(AstChild(name, MeshAxisPattern(), axis, "mesh_axis", "mesh_axis"))
        states.append((name, kind, reduction))
    return dataclasses.replace(
        matched,
        children=tuple(children),
        captures={
            **matched.captures,
            "states": tuple(states),
        },
    )


class StatedLayoutPattern(ElementPattern):
    element_name = "stated_layout"
    syntax = LazyPattern(
        lambda: BindPattern(
            AstNodePattern(
                ast.BinOp,
                FieldPattern("op", AstNodePattern(ast.MatMult)),
                FieldPattern(
                    "left",
                    ChildPattern(
                        "operand",
                        ChoicePattern(
                            ComposedLayoutSugarPattern(),
                            PlacementOperandPattern(),
                            StaticValuePattern(),
                        ),
                        "layout",
                        "layout",
                    ),
                ),
                FieldPattern("right", ValueStatesPattern()),
            ),
            bind_stated_layout,
        )
    )

    def construct(self, match, children, context):
        states = tuple(
            (*children[name], kind, reduction) for name, kind, reduction in match.captures["states"]
        )
        return children["operand"], states

    RULES = (StatedLayoutOperandRule(), *PlacedLayoutPattern.RULES)


@dataclass(frozen=True)
class ComposedLayoutOffsetRule:
    STATEMENT: ClassVar[str] = (
        "A composed layout has one offset before its inner layout; "
        "compound offsets are parenthesized."
    )

    def apply(self, value, *, node, context):
        outer = node
        if isinstance(outer.op, ast.BitOr):
            inner = outer.right
            outer = outer.left
            if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Add):
                return PatternFailure(
                    "composed_layout", inner, "offset belongs before '|': write L + off | inner"
                )
        if isinstance(outer, ast.BinOp) and isinstance(outer.op, ast.Add):
            outer = outer.left
            if isinstance(outer, ast.BinOp) and isinstance(outer.op, ast.Add):
                return PatternFailure("composed_layout", outer, "write one offset: L + (a + b)")
        return value


@dataclass(frozen=True)
class ComposedLayoutOuterRule:
    STATEMENT: ClassVar[str] = "A composed layout's outer is a plain Layout without mesh placement."

    def apply(self, value, *, match, context):
        outer = value["outer"]
        if isinstance(outer, PlacedLayout):
            outer = outer.layout
        if not isinstance(outer, types.Layout):
            raise ParseError.from_node(
                match.node, context, "composed layout outer must be a Layout without mesh placement"
            )
        return {**value, "outer": outer}


@dataclass(frozen=True)
class ComposedLayoutInnerRule:
    STATEMENT: ClassVar[str] = "A composed layout's inner is a Swizzle or LayoutBase."

    def apply(self, value, *, match, context):
        inner = value.get("inner")
        if "inner" in value and not isinstance(inner, (types.Swizzle, runtime.LayoutBase)):
            raise ParseError.from_node(
                match.node, context,
                f"composed layout inner must be a Swizzle or a layout, got {type(inner).__name__}",
            )
        return value


@dataclass(frozen=True)
class ComposedLayoutConstructionRule:
    STATEMENT: ClassVar[str] = "A checked composition constructs its inner, offset, and outer layout."

    def apply(self, value, *, match, context):
        inner, offset = value.get("inner"), value.get("offset", 0)
        if inner is None and offset == 0:
            return value["outer"]
        return types.ComposedLayout(inner, offset, value["outer"])


def composed_outer_syntax():
    return ChildPattern(
        "outer",
        ChoicePattern(PlacedLayoutPattern(), PlainLayoutPattern(), StaticValuePattern()),
        "mesh_layout",
        "layout",
    )


def composed_offset_syntax():
    return AstNodePattern(
        ast.BinOp,
        FieldPattern("left", composed_outer_syntax()),
        FieldPattern("op", AstNodePattern(ast.Add)),
        FieldPattern("right", ChildPattern("offset", DimExprPattern(), "layout_offset")),
    )


def bind_composed_layout_sugar(node, context, matched):
    for rule in ComposedLayoutSugarPattern.BIND_RULES:
        matched = rule.apply(matched, node=node, context=context)
        if isinstance(matched, MatchFailure):
            return matched
    return dataclasses.replace(
        matched,
        branch_id="composed_layout",
        pattern_id="tensor.layout.composed",
    )


class ComposedLayoutSugarPattern(ElementPattern):
    """Compose an unplaced layout only in a layout grammar position."""

    element_name = "composed_layout"
    syntax = LazyPattern(
        lambda: BindPattern(
            ChoicePattern(
                composed_offset_syntax(),
                AstNodePattern(
                    ast.BinOp,
                    FieldPattern(
                        "left",
                        ChoicePattern(
                            composed_offset_syntax(),
                            composed_outer_syntax(),
                        ),
                    ),
                    FieldPattern("op", AstNodePattern(ast.BitOr)),
                    FieldPattern(
                        "right", ChildPattern("inner", StaticValuePattern(), "static_layout")
                    ),
                ),
            ),
            bind_composed_layout_sugar,
        )
    )

    def construct(self, match, children, context):
        return dict(children)

    BIND_RULES = (ComposedLayoutOffsetRule(),)
    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        ComposedLayoutOuterRule(), ComposedLayoutInnerRule(), ComposedLayoutConstructionRule(),
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
            StatedLayoutPattern(),
            PlacedLayoutPattern(),
            PlainLayoutPattern(),
            ComposedLayoutSugarPattern(),
        )
    )

    def construct(self, match, children, context):
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

    def construct(self, match, children, context):
        if isinstance(match.node, ast.Name) and match.node.id in {
            str(k) for k in runtime.StorageKind
        }:
            raw = match.node.id
        elif isinstance(match.node, ast.Constant):
            raw = match.node.value
        else:
            raw = resolve_reference(match.node, context)
        try:
            return runtime.resolve_storage(raw)
        except (TypeError, ValueError) as error:
            raise ParseError.from_node(match.node, context, str(error)) from error

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (StorageValueRule(),)


def slot_kind(node: object, context: MatchContext) -> str | None:
    if isinstance(node, ast.Constant):
        if node.value is None:
            return "layout"
        if isinstance(node.value, str):
            return "storage"
    if isinstance(node, (ast.Tuple, ast.Call, ast.BinOp)):
        return "layout"
    if isinstance(node, (ast.Name, ast.Attribute)):
        try:
            value = resolve_reference(node, context)
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
        )
    )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


def states_a_layout_slot(elts: object, context: "MatchContext") -> bool:
    """Whether an annotation's third slot states a layout rather than a storage.

    Read before the sequence is matched, because a child is not parsed until the
    branch is already chosen: two branches of the same arity would otherwise be
    settled by their order alone. A slot nobody can classify is left to the
    storage branch, whose own patterns report it.
    """
    if not isinstance(elts, (tuple, list)) or len(elts) < 3:
        return False
    try:
        return slot_kind(elts[2], context) == "layout"
    except ParseError:
        return False


class TensorShapeLayoutPattern(ElementPattern):
    element_name = "tensor_shape_layout"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            StatedLayoutPattern(),
            PlacedLayoutPattern(),
            ShapePattern(),
        )
    )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


def tensor_head(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


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
                            lambda node, context: tensor_head(node) in {"Tensor", "ConstTensor"},
                        ),
                        CapturePattern(
                            "head",
                            lambda node, context: tensor_head(node),
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
                                ConditionPattern(
                                    "third slot states a layout",
                                    states_a_layout_slot,
                                    SequencePattern(
                                        ChildPattern(
                                            "shape",
                                            lambda: ShapePattern(),
                                            "tensor_shape",
                                            "tensor_shape",
                                        ),
                                        ChildPattern(
                                            "dtype",
                                            lambda: DTypePattern(),
                                            "tensor_dtype",
                                            "dtype",
                                        ),
                                        ChildPattern(
                                            "layout",
                                            lambda: TensorOptionalSlotPattern(),
                                            "tensor_optional_slot",
                                            "layout",
                                        ),
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
                                        "storage",
                                        lambda: TensorOptionalSlotPattern(),
                                        "tensor_optional_slot",
                                        "storage",
                                    ),
                                ),
                                SequencePattern(
                                    ChildPattern(
                                        "shape",
                                        lambda: ShapePattern(),
                                        "tensor_shape",
                                        "tensor_shape",
                                    ),
                                    ChildPattern(
                                        "dtype",
                                        lambda: DTypePattern(),
                                        "tensor_dtype",
                                        "dtype",
                                    ),
                                    ChildPattern(
                                        "layout",
                                        lambda: TensorOptionalSlotPattern(),
                                        "tensor_optional_slot",
                                        "layout",
                                    ),
                                    ChildPattern(
                                        "storage",
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

    def child_values(self, match, child, children, context):
        storage = next((c for c in match.children if c.name == "storage"), None)
        return {
            "layout_storage": runtime.StorageKind.GMEM
            if storage is None
            else parse_node(
                storage.pattern,
                storage.node,
                context.child(situation=storage.situation, role=storage.role),
            )
        }

    def construct(self, match, children, context):
        """One layout, from one slot.

        A shape slot's placement sugar states a layout, and so does a layout
        slot. Accepting both at once meant one silently replaced the other, so
        the arity branches keep them apart: a stated layout slot takes a plain
        shape beside it.
        """
        placed = children.get("shape_or_layout")
        if placed is not None:
            if isinstance(placed, PlacedLayout):
                shape, layout = placed.shape, placed.layout
            else:
                shape, layout = placed, None
        else:
            shape, layout = children["shape"], children["layout"]
        if isinstance(layout, PlacedLayout):
            layout = layout.layout
        storage = children.get("storage") or runtime.StorageKind.GMEM
        type_ = runtime.TensorType(
            shape=shape,
            dtype=children["dtype"],
            layout=types.normalize_layout(layout) if layout is not None else None,
            storage=storage,
        )
        if layout is None and context.values.get("position") not in {
            "hir_input",
            "hir_output",
            "tir_input",
            "tir_output",
        }:
            return dataclasses.replace(type_, layout=runtime.Layout(shape))
        return type_

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        TensorLayoutStorageRule(),
        TensorPositionRule(),
    )

class ScalarTypePattern(ElementPattern):
    element_name = "scalar_type"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            BranchPattern(
                "type_reference",
                ReferencePattern(),
                pattern_id="type.reference",
            ),
            BranchPattern(
                "type_call",
                ChildPattern(
                    "value",
                    lambda: StaticCallPattern(),
                    "type_constructor",
                ),
                pattern_id="type.call",
            ),
        )
    )

    def construct(self, match, children, context):
        value = (
            children["value"]
            if match.branch_id == "type_call"
            else resolve_reference(match.node, context)
        )
        if not isinstance(
            value,
            (runtime.TensorType, runtime.TupleType, runtime.UnitType, runtime.PointerType, runtime.CommunicationHandleType),
        ):
            raise ParseError.from_node(match.node, context, "annotation did not resolve to IR Type")
        return value

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class TupleTypePattern(ElementPattern):
    """Parse ``tuple[T, U, ...]`` as a parser-only type annotation value."""

    element_name = "tuple_type"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            BranchPattern(
                "tuple",
                AstNodePattern(
                    ast.Subscript,
                    FieldPattern(
                        "value",
                        AstNodePattern(
                            ast.Name,
                            FieldPattern("id", LiteralPattern("tuple")),
                        ),
                    ),
                    FieldPattern(
                        "slice",
                        AstNodePattern(
                            ast.Tuple,
                            CapturePattern("field_count", lambda node, context: len(node.elts)),
                            FieldPattern(
                                "elts",
                                RepeatPattern(
                                    ChildPattern(
                                        "field_{index}",
                                        lambda: TypeAnnotationPattern(),
                                        "type_annotation",
                                        "tuple_field",
                                    ),
                                    minimum=1,
                                ),
                            ),
                        ),
                    ),
                ),
                pattern_id="tuple.annotation",
            ),
            BranchPattern(
                "tuple_single",
                AstNodePattern(
                    ast.Subscript,
                    FieldPattern(
                        "value",
                        AstNodePattern(
                            ast.Name,
                            FieldPattern("id", LiteralPattern("tuple")),
                        ),
                    ),
                    FieldPattern(
                        "slice",
                        ChildPattern(
                            "field_0",
                            lambda: TypeAnnotationPattern(),
                            "type_annotation",
                            "tuple_field",
                        ),
                    ),
                ),
                pattern_id="tuple.annotation",
            ),
        )
    )

    def construct(self, match, children, context):
        fields = tuple(
            children[f"field_{index}"] for index in range(match.captures.get("field_count", 1))
        )
        if not all(isinstance(field, (runtime.TensorType, runtime.TupleType)) for field in fields):
            raise ParseError.from_node(
                match.node,
                context,
                "tuple annotation fields must resolve to Tensor or tuple types",
            )
        return runtime.TupleType(fields=fields)

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class TypeAnnotationPattern(ElementPattern):
    element_name = "type_annotation"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            TensorPattern(),
            TupleTypePattern(),
            ScalarTypePattern(),
        )
    )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


def constraint_value(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    raise ValueError(
        f"where layout extent must be a literal or symbolic name, got {type(node).__name__}"
    )


def parse_partial_constraint(node: ast.AST):
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError('Partial binding must use P("reduction")')
    if node.func.id != "P" or len(node.args) != 1 or node.keywords:
        raise ValueError('Partial binding must use P("reduction")')
    value = constraint_value(node.args[0])
    if not isinstance(value, str) or not value:
        raise ValueError("partial reduction must be a non-empty string")
    return types.Partial(value)


@dataclass(frozen=True)
class WhereTopologyStatesRule:
    STATEMENT: ClassVar[str] = (
        "Where layout states constrain named topologies with topology @ B() or "
        "topology @ P(nonempty string); they do not bind lexical mesh axes."
    )

    def apply(self, value, *, node, context):
        if not isinstance(node, ast.Set) or not node.elts:
            raise ValueError("layout bindings must be a nonempty set")
        bindings = []
        for item in node.elts:
            if not isinstance(item, ast.BinOp) or not isinstance(item.op, ast.MatMult):
                raise ValueError("layout bindings must use `topology @ B()` or `P()`")
            topology = constraint_value(item.left)
            if not isinstance(topology, str) or not topology:
                raise ValueError("layout binding topology must be symbolic")
            if (
                isinstance(item.right, ast.Call)
                and isinstance(item.right.func, ast.Name)
                and item.right.func.id == "B"
                and not item.right.args
                and not item.right.keywords
            ):
                attribute = types.Broadcast()
            else:
                attribute = parse_partial_constraint(item.right)
            bindings.append((topology, attribute))
        return bindings


def parse_constraint_bindings(node: ast.AST, context: MatchContext):
    value = None
    for rule in WhereAnnotationPattern.BIND_RULES:
        value = rule.apply(value, node=node, context=context)
    return value


def where_static(node: ast.AST, context: MatchContext):
    if isinstance(node, ast.Constant):
        return node.value
    return resolve_reference(node, context)


def parse_layout_constraint(node: ast.AST, context: MatchContext):
    states_node = None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        states_node = node.right
        node = node.left
    if not isinstance(node, ast.Tuple):
        raise ValueError("layout constraint must be a tuple")
    dims_node = node
    if not dims_node.elts:
        raise ValueError("layout constraint cannot be empty")

    def resolve_extent(item: ast.AST):
        value = where_static(item, context)
        if isinstance(value, bool) or not isinstance(value, (int, expr.DimVar)):
            raise ValueError("layout dimensions must use `_`, an integer, or a symbolic extent")
        return value

    shape = []
    bindings = []
    for index, item in enumerate(dims_node.elts):
        if isinstance(item, ast.Name) and item.id == "_":
            shape.append(clause.LAYOUT_WILDCARD)
            continue
        if isinstance(item, ast.Name) and item.id == "D":
            raise ValueError("layout broadcast must use `layout @ {topology @ B()}`")
        if isinstance(item, ast.BinOp) and isinstance(item.op, ast.MatMult):
            extent = resolve_extent(item.left)
            topology = constraint_value(item.right)
            if not isinstance(topology, str) or not topology:
                raise ValueError("layout topology binding must be symbolic")
            shape.append(extent)
            bindings.append((topology, types.Split(index)))
            continue
        shape.append(resolve_extent(item))
    if states_node is not None:
        bindings.extend(parse_constraint_bindings(states_node, context))
    if len({topology for topology, _ in bindings}) != len(bindings):
        raise ValueError("layout constraint cannot bind one topology more than once")
    return clause.Layout(layout=types.Layout(shape=tuple(shape)), bindings=tuple(bindings))


class WhereAnnotationPattern(ElementPattern):
    element_name = "where_annotation"
    BIND_RULES = (WhereTopologyStatesRule(),)
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

    def construct(self, match, children, context):
        node = match.node
        source_range = authored_source_range(node, context)
        location = (
            clause.SourceLocation(*source_range)
            if source_range is not None
            else clause.SourceLocation()
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
                            parse_layout_constraint(keyword.value, context),
                            source_loc=location,
                            provenance=clause.Provenance.AUTHOR,
                        )
                    )
                elif keyword.arg == "mesh":
                    constraints.append(
                        clause.Mesh(
                            mesh=where_static(keyword.value, context),
                            source_loc=location,
                            provenance=clause.Provenance.AUTHOR,
                        )
                    )
                elif keyword.arg == "storage":
                    constraints.append(
                        clause.Storage(
                            storage=where_static(keyword.value, context),
                            source_loc=location,
                            provenance=clause.Provenance.AUTHOR,
                        )
                    )
                else:
                    raise ValueError(
                        f"where(...) has unknown field {keyword.arg!r}; use "
                        "layout=..., mesh=..., or storage=..."
                    )
        except (TypeError, ValueError) as error:
            raise ParseError.from_node(node, context, str(error)) from error
        return clause.WhereMetadata(constraints=tuple(constraints), source_loc=location)

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

    def construct(self, match, children, context):
        return children["type"]

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


def enter_signature(
    node: object, context: MatchContext, matched: AstMatch[Any]
) -> AstMatch[Any] | None:
    assert isinstance(node, ast.arguments)
    if context.function is None:
        return None
    names = tuple(argument.arg for argument in node.args)
    constness = tuple(
        isinstance(argument.annotation, ast.Subscript)
        and tensor_head(argument.annotation.value) == "ConstTensor"
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
            enter_signature,
        )
    )

    def construct(self, match, children, context):
        names = match.captures["names"]
        constness = match.captures["constness"]
        params = tuple(
            runtime.Var(
                type=runtime.bind_annotation(children[f"parameter_{index}"]),
                annotation=children[f"parameter_{index}"],
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

    def construct(self, match, children, context):
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

    def construct(self, match, children, context):
        if match.branch_id == "static_name":
            return resolve_reference(match.node, context)
        elif match.branch_id == "static_attribute":
            owner = children["owner"]
            attribute = match.captures["attribute"]
            if is_slot_value(owner):
                return resolve_reference(match.node, context)
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


def static_branch(node_type: type, constructor: type, pattern_id: str) -> AstPattern[Any]:
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


class StaticSequencePattern(ElementPattern):
    element_name = "sequence"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            static_branch(ast.Tuple, tuple, "static.tuple"),
            static_branch(ast.List, list, "static.list"),
            static_branch(ast.Set, set, "static.set"),
        )
    )

    def construct(self, match, children, context):
        return match.captures["constructor"](children.values())

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


def enter_static_dict(
    node: object, context: MatchContext, matched: AstMatch[Any]
) -> AstMatch[Any] | MatchFailure | None:
    assert isinstance(node, ast.Dict)
    if any(key is None for key in node.keys):
        return PatternFailure("static_dict", node, "`**` unpacking is not a static dictionary form")
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


class StaticDictPattern(ElementPattern):
    element_name = "dict"
    syntax = LazyPattern(
        lambda: BindPattern(
            AstNodePattern(
                ast.Dict,
                FieldPattern("keys", RepeatPattern(StaticValuePattern())),
                FieldPattern("values", RepeatPattern(StaticValuePattern())),
            ),
            enter_static_dict,
        )
    )

    def construct(self, match, children, context):
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
                CapturePattern("operator", lambda node, context: BINARY_OPERATORS[type(node.op)]),
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

    def construct(self, match, children, context):
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
                CapturePattern("operator", lambda node, context: UNARY_OPERATORS[type(node.op)]),
                FieldPattern(
                    "operand",
                    ChildPattern("operand", StaticValuePattern(), "static_operand"),
                ),
            ),
            pattern_id="static.unary",
        )
    )

    def construct(self, match, children, context):
        try:
            return match.captures["operator"](children["operand"])
        except (TypeError, ValueError) as error:
            raise ParseError.from_node(match.node, context, str(error)) from error

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


@dataclass(frozen=True)
class InlineMeshCallRule:
    STATEMENT: ClassVar[str] = (
        "Inside an open mesh scope, Mesh constructors open a with header unless used as a ShardLayout mesh value."
    )

    def apply(self, value, *, match, context):
        if (
            value["callee"] is runtime.Mesh
            and not context.values.get("shard_layout_mesh", False)
            and context.function is not None
            and context.function.state.mesh_stack
        ):
            raise ParseError.from_node(
                match.node, context, "open meshes with `with Mesh(...) as name`"
            )
        return value


@dataclass(frozen=True)
class StaticCallConstructionRule:
    STATEMENT: ClassVar[str] = "A static call has a callable target and constructs a valid Python value."

    def apply(self, value, *, match, context):
        callee = value["callee"]
        if not callable(callee) and not isinstance(callee, type):
            raise ParseError.from_node(match.node, context, "static calls require a callable target")
        args = tuple(value[f"arg_{index}"] for index in range(match.captures["arg_count"]))
        kwargs = {name: value[f"kw_{name}"] for name in match.captures["keywords"]}
        try:
            if match.captures.get("default_shard_modes"):
                arguments = inspect.signature(callee).bind(*args, **kwargs).arguments
                target = None if context.function is None else context.function.target
                return runtime.ShardLayout.from_modes(
                    arguments["layout"],
                    arguments["mesh"],
                    arguments["attrs"],
                    storage=context.values.get("layout_storage", runtime.StorageKind.GMEM),
                    owners=target.storage_owners if target else expr.StorageOwners((), {}),
                )
            return callee(*args, **kwargs)
        except (TypeError, ValueError) as error:
            raise ParseError.from_node(match.node, context, str(error)) from error


def is_ceildiv(node, context):
    try:
        return (
            isinstance(node, (ast.Name, ast.Attribute))
            and resolve_reference(node, context) is dim.ceildiv
        )
    except ParseError:
        return False


class CeildivExpressionPattern(ElementPattern):
    element_name = "ceildiv_expression"
    syntax = LazyPattern(
        lambda: AstNodePattern(
            ast.Call,
            FieldPattern(
                "func",
                PredicatePattern(
                    "ceildiv",
                    is_ceildiv,
                ),
            ),
            FieldPattern(
                "args",
                SequencePattern(
                    ChildPattern("left", IndexEndpointPattern(), "slice_endpoint"),
                    ChildPattern("right", IndexEndpointPattern(), "slice_endpoint"),
                ),
            ),
            FieldPattern("keywords", SequencePattern()),
        )
    )

    def construct(self, match, children, context):
        return dim.ceildiv(children["left"], children["right"])

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


def enter_static_call(
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
    shard_layout = (
        isinstance(node.func, (ast.Name, ast.Attribute))
        and resolve_reference(node.func, context) is runtime.ShardLayout
    )
    default_shard_modes = False

    def argument_node(argument, is_layout):
        nonlocal default_shard_modes
        if not (shard_layout and is_layout and isinstance(argument, ast.Call)):
            return argument
        if resolve_reference(argument.func, context) is not runtime.Layout:
            return argument
        try:
            arguments = (
                inspect.signature(runtime.Layout)
                .bind(
                    *argument.args,
                    **{keyword.arg: keyword.value for keyword in argument.keywords},
                )
                .arguments
            )
        except TypeError as error:
            raise ParseError.from_node(argument, context, str(error)) from error
        strides = arguments.get("strides")
        if strides is None or isinstance(strides, ast.Constant) and strides.value is None:
            default_shard_modes = True
            return arguments["shape"]
        return argument

    children = [AstChild("callee", StaticValuePattern(), node.func, "static_callee")]
    children.extend(
        AstChild(
            f"arg_{index}",
            StaticValuePattern(),
            argument_node(argument, index == 0),
            "static_argument",
            values={"shard_layout_mesh": shard_layout and index == 2},
        )
        for index, argument in enumerate(node.args)
    )
    children.extend(
        AstChild(
            f"kw_{keyword.arg}",
            StaticValuePattern(),
            argument_node(keyword.value, keyword.arg == "layout"),
            "static_argument",
            keyword.arg,
            values={"shard_layout_mesh": shard_layout and keyword.arg == "mesh"},
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
            "default_shard_modes": default_shard_modes,
        },
        children=tuple(children),
    )


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
            enter_static_call,
        )
    )

    def construct(self, match, children, context):
        return children

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        InlineMeshCallRule(), StaticCallConstructionRule(),
    )


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

    def construct(self, match, children, context):
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

    def construct(self, match, children, context):
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

    def construct(self, match, children, context):
        name = match.captures["name"]
        value = context.lexical_scope.lookup(name)
        if value is None:
            try:
                value = resolve_reference(match.node, context)
            except ParseError:
                pass
        if isinstance(value, slice):
            value = value.start
        if isinstance(value, runtime.Expr):
            return value
        if isinstance(value, (bool, int, float)):
            return constant(value)
        raise ParseError.from_node(match.node, context, f"name {name!r} is not an Expr")

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class ConstantPattern(ElementPattern):
    element_name = "constant"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            BranchPattern(
                "constant",
                AstNodePattern(
                    ast.Constant,
                    FieldPattern("value", LiteralPattern(value_type=(bool, int, float))),
                ),
                pattern_id="expression.constant",
            ),
            BranchPattern(
                "signed_constant",
                AstNodePattern(
                    ast.UnaryOp,
                    FieldPattern("op", AstNodePattern(ast.USub)),
                    FieldPattern(
                        "operand",
                        AstNodePattern(
                            ast.Constant,
                            FieldPattern(
                                "value",
                                PredicatePattern(
                                    "numeric-literal",
                                    lambda value, context: type(value) in (int, float),
                                ),
                            ),
                        ),
                    ),
                    CapturePattern("value", lambda node, context: ast.literal_eval(node)),
                ),
                pattern_id="expression.signed_constant",
            ),
        )
    )

    def construct(self, match, children, context):
        return constant(match.captures["value"])

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

    def construct(self, match, children, context):
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
        return value


@dataclass(frozen=True)
class AuthoredCallScopeRule:
    STATEMENT: ClassVar[str] = "An authored HIR call must run inside a mesh if its function opens one."

    def apply(self, value, *, match, context):
        note_authored_call(match.node, context)
        return value


@dataclass(frozen=True)
class CallTypeInferenceRule:
    STATEMENT: ClassVar[str] = "A call's result type must be inferred from its binding."

    def apply(self, value, *, match, context):
        if not isinstance(value, runtime.Call):
            return value
        infer_context = context.lexical_scope.lookup(TYPE_INFER_CONTEXT)
        if not isinstance(infer_context, runtime.TypeInferContext):
            infer_context = runtime.TypeInferContext()
        if context.function.dialect == "tir" and isinstance(value.target, runtime.FunctionRef):
            return value
        computed = runtime.TypeInferVisitor().visit(value, infer_context)
        value.retype(computed)
        return value


@dataclass(frozen=True)
class VariadicInputs:
    items: tuple[runtime.Expr, ...]

    def __init__(self, values):
        items = tuple(values)
        if not all(isinstance(value, runtime.Expr) for value in items):
            raise TypeError("variadic inputs must contain Expr values")
        object.__setattr__(self, "items", items)


def enter_variadic_sequence(
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


def enter_variadic_comprehension(
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
                    enter_variadic_sequence,
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
                    enter_variadic_sequence,
                ),
                pattern_id="call.variadic.tuple",
            ),
            BindPattern(
                AstNodePattern(ast.ListComp),
                enter_variadic_comprehension,
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

    def construct(self, match, children, context):
        return VariadicInputs(tuple(children.values()))

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


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
        if len(inputs) != 1 or expr.variadic_item_annotation(inputs[0]) is None:
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


@dataclass(frozen=True)
class OpValueKeywordOnlyRule:
    STATEMENT: ClassVar[str] = "An op value takes only keyword attributes."

    def apply(self, value, *, node, context):
        if node.args:
            return PatternFailure("op_value", node, "op attributes take keyword arguments")
        return value


@dataclass(frozen=True)
class OpValueAttributeRule:
    STATEMENT: ClassVar[str] = "Each op-value keyword names an attribute of its resolved op schema."

    def apply(self, value, *, node, context):
        attrs = {param.name for param in value.captures["schema"].signature if param.kind == "attribute"}
        if node.arg not in attrs:
            return PatternFailure(
                "op_value", value.node,
                f"{ast.unparse(value.node.func)} has no attribute {node.arg!r}"
            )
        return value


@dataclass(frozen=True)
class OpValueUniqueKeywordRule:
    STATEMENT: ClassVar[str] = "An op-value keyword is supplied at most once."

    def apply(self, value, *, node, context):
        if node.arg in value.captures["bound_keywords"]:
            return PatternFailure(
                "op_value", value.node, f"keyword given more than once: {node.arg}"
            )
        value.captures["bound_keywords"].add(node.arg)
        return value


@dataclass(frozen=True)
class OpValueConstructionRule:
    STATEMENT: ClassVar[str] = "An op value constructs valid attributes using its resolved schema."

    def apply(self, value, *, match, context):
        try:
            attrs = {
                name: item.layout if isinstance(item, PlacedLayout) else item
                for name, item in value.items()
            }
            return match.captures["schema"].builder(**attrs)
        except (TypeError, ValueError) as error:
            raise ParseError.from_node(match.node, context, str(error)) from error


def bind_op_value(node, context, matched):
    try:
        callee = resolve_reference(node.func, context)
    except ParseError:
        return None
    schema = callee if isinstance(callee, runtime.OpSchema) else expr.schema_of(callee)
    if not isinstance(schema, runtime.OpSchema):
        return None
    matched = dataclasses.replace(
        matched, captures={**matched.captures, "schema": schema, "bound_keywords": set()}
    )
    matched = OpValuePattern.BIND_RULES[0].apply(matched, node=node, context=context)
    if isinstance(matched, MatchFailure):
        return matched
    attrs = {param.name: param for param in schema.signature if param.kind == "attribute"}
    children = []
    for keyword in node.keywords:
        for rule in OpValuePattern.BIND_RULES[1:]:
            matched = rule.apply(matched, node=keyword, context=context)
            if isinstance(matched, MatchFailure):
                return matched
        param = attrs[keyword.arg]
        children.append(
            AstChild(
                keyword.arg,
                pattern_for_param(param, keyword.value),
                keyword.value,
                "call_attribute",
                param.name,
            )
        )
    return dataclasses.replace(
        matched,
        pattern_id="call.op_value",
        branch_id="op_value",
        captures={**matched.captures, "schema": schema},
        children=tuple(children),
    )


class OpValuePattern(ElementPattern):
    """Construct an op attribute with its own schema's attribute grammar."""

    element_name = "op_value"
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
                                "arg", CapturePattern("keyword_name", lambda value, context: value)
                            ),
                            FieldPattern("value", AstNodePattern(ast.expr)),
                        )
                    ),
                ),
            ),
            bind_op_value,
        )
    )

    def match(self, node, context):
        matched = super().match(node, context)
        return StaticValuePattern().match(node, context) if matched is None else matched

    def construct(self, match, children, context):
        return children

    BIND_RULES = (
        OpValueKeywordOnlyRule(), OpValueAttributeRule(), OpValueUniqueKeywordRule(),
    )
    RULES: ClassVar[tuple[AstRule[Any], ...]] = (OpValueConstructionRule(),)


def pattern_for_param(param: object, node: ast.AST) -> AstPattern[Any]:
    annotation = param.annotation
    if expr.variadic_item_annotation(param) is not None:
        kind = "variadic"
    elif annotation is runtime.TensorType and isinstance(node, ast.Subscript):
        kind = "tensor"
    elif annotation is runtime.DType:
        kind = "dtype"
    elif annotation is runtime.StorageKind or param.name == "storage":
        kind = "storage"
    elif annotation in (runtime.Layout, runtime.ShardLayout, runtime.LayoutBase):
        kind = "layout"
    elif annotation is expr.Op and isinstance(node, ast.Call):
        kind = "op"
    else:
        kind = "static"
    return CallPattern.PARAM_PATTERNS[kind]()


def schema_children(
    node: ast.Call, schema: object, context: MatchContext
) -> tuple[AstChild, ...] | MatchFailure | None:
    """Bind a call's arguments to one op schema's inputs and attributes.

    A ``Tuple[T]`` input consumes exactly one explicit sequence and flattens
    its elements into ``Call.args``. Attributes remain keyword-only, so the
    sequence boundary cannot be confused with an attribute position.

    An input declared ``optional`` may be left off the end of the call. A
    resource operand such as ``reduce``'s or ``dot``'s workspace is present
    only in the form that needs one, so counting it as required would make
    the other form unauthorable.
    """
    params = tuple(schema.signature)
    inputs = [param for param in params if param.kind == "input"]
    attrs = [param for param in params if param.kind == "attribute"]
    variadic = len(inputs) == 1 and expr.variadic_item_annotation(inputs[0]) is not None
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
                pattern_for_param(inputs[0], positional[0]),
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
                pattern_for_param(param, argument),
                argument,
                "call_attribute",
                "allocation" if param.annotation is runtime.TensorType else param.name,
            )
        )
    for keyword in node.keywords:
        if keyword.arg is None:
            return PatternFailure("op_call", node, "`**` unpacking is not an authored call form")
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
                pattern_for_param(param, keyword.value),
                keyword.value,
                "call_attribute",
                "allocation" if param.annotation is runtime.TensorType else param.name,
            )
        )
    required = [param for param in inputs if not param.optional]
    if not variadic and len(positional) < len(required):
        return PatternFailure(
            "op_call",
            node,
            f"{schema.name} takes {len(required)} inputs, got {len(positional)}",
        )
    return tuple(children)


def folded_index(index) -> int | None:
    """An authored slot subscript as an integer path segment, when it is a constant integer.

    A ``Constant`` folds only when its type is an integer scalar; any other one
    stays a run-time operand for the index check to refuse.
    """
    if isinstance(index, runtime.Constant):
        if not core.is_index_type(index.type):
            return None
        index = index.value
    return index if isinstance(index, int) and not isinstance(index, bool) else None


def slot_binding(node, context: MatchContext) -> str | None:
    """The name of this Module's class-body slot that *node* spells, if any."""
    module = context.function.module
    if not isinstance(node, ast.Name) or module is None or node.id not in module.children:
        return None
    try:
        value = resolve_reference(node, context)
    except ParseError:
        return None
    return node.id if value is module.children[node.id] else None


def is_slot_value(value) -> bool:
    """Whether *value* is a Module definition, a slot wrapper, or a tuple of definitions."""
    return isinstance(value, (runtime.Module, runtime.ModuleList)) or (
        isinstance(value, tuple) and bool(value) and all(isinstance(m, runtime.Module) for m in value)
    )


def slot_reference(node, context: MatchContext):
    """Split a callee written from a slot of this Module into its root slot and steps.

    The steps are ``("attr", name)`` and ``("index", slice)`` from left to right.
    Anything not rooted at a slot of this Module is not a slot reference.
    """
    steps = []
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        steps.append(("attr", node.attr) if isinstance(node, ast.Attribute) else ("index", node.slice))
        node = node.value
    root = slot_binding(node, context)
    return None if root is None else (root, tuple(reversed(steps)))


def reference_index_children(steps) -> tuple[AstChild, ...]:
    """One subscript-index child per index step of a slot reference."""
    indices = [value for kind, value in steps if kind == "index"]
    return tuple(
        AstChild(f"ref_index_{k}", SubscriptIndexPattern(), index, "subscript_index", "subscript_index")
        for k, index in enumerate(indices)
    )


def build_slot_reference(match, children, context, *, launch=False):
    """The reference a slot reference spells, its run-time index operands, and its function.

    Integer scalar constants fold into the path; any other index stays a
    leading operand that must be an integer scalar. Steps that end at a function
    name name it; a call through subscripts alone names the instance entry. A
    launch must name its function and refuses a run-time index. The function
    comes from ``core.resolve`` in the declaring Module, given the complete
    reference.
    """
    root, steps = match.captures["reference"]
    path, operands, given = [root], [], iter(children[f"ref_index_{k}"] for k in range(len(steps)))
    for kind, value in steps:
        if kind == "attr":
            path.append(value)
            continue
        index = next(given)
        folded = folded_index(index)
        path.append(folded)
        if folded is None:
            operands.append(constant(index) if isinstance(index, (int, float)) else index)
    try:
        if launch and operands:
            raise ValueError(f"launch {runtime.FunctionRef(tuple(path), None).qualified_name!r}: a launch index must be a constant")
        named = bool(steps) and steps[-1][0] == "attr"
        if not named and (launch or any(kind == "attr" for kind, _ in steps)):
            raise ValueError(
                f"{runtime.FunctionRef(tuple(path), None).qualified_name!r}: name the function to call"
            )
        ref = runtime.FunctionRef(tuple(path[:-1]), path[-1]) if named else runtime.FunctionRef(tuple(path), None)
        ref.check_indices(tuple(getattr(operand, "type", None) for operand in operands))
        _, function = core.resolve(context.function.module, ref)
    except ValueError as error:
        raise ParseError.from_node(match.node, context, str(error)) from error
    return ref, tuple(operands), function


def declared_prototype(callee, node, context: MatchContext):
    """The prototype of this Module that *callee* is or specializes, already parsed.

    A call names a sibling by its prototype; dispatch picks the variant. A bare
    function has no Module to name a sibling in, and a function from outside
    this Module is not a sibling.
    """
    module = context.function.module
    if module is None:
        return PatternFailure(
            "op_call", node, f"a bare function cannot call function {callee.name!r}; declare both in a @module"
        )
    for prototype in module.functions:
        if callee is prototype or any(callee is variant for variant in prototype.variants):
            return prototype
    return PatternFailure(
        "op_call",
        node,
        f"function {callee.name!r} is not declared before this call in @module {module.owner_name!r}",
    )


def enter_call(
    node: object, context: MatchContext, matched: AstMatch[Any]
) -> AstMatch[Any] | MatchFailure | None:
    assert isinstance(node, ast.Call)
    if reference := slot_reference(node.func, context):
        if context.function.dialect != "tir" and any(kind == "attr" for kind, _ in reference[1]):
            return PatternFailure(
                "op_call", node.func, "only calling an instance entry is supported"
            )
        if node.keywords:
            return PatternFailure(
                "function_ref_call", node, "a function call takes positional arguments only"
            )
        return dataclasses.replace(
            matched,
            pattern_id="call.symbol",
            branch_id="function_ref_call",
            captures={**matched.captures, "reference": reference},
            children=(
                *reference_index_children(reference[1]),
                *(
                    AstChild(f"arg_{index}", ExpressionPattern(), argument, "call_argument", "argument")
                    for index, argument in enumerate(node.args)
                ),
            ),
        )
    subscript = isinstance(node.func, ast.Subscript)
    try:
        callee = resolve_reference(node.func.value if subscript else node.func, context)
    except DeclarationOrderError:
        raise
    except ParseError:
        return None
    if is_slot_value(callee):
        return PatternFailure(
            "op_call",
            node.func,
            "instance calls must use a slot name in the @module class body",
        )
    if subscript:
        return None
    if isinstance(callee, (runtime.Function, runtime.PrimFunction)):
        if node.keywords:
            return PatternFailure(
                "op_call", node, "a function call takes positional arguments only"
            )
        callee = declared_prototype(callee, node.func, context)
        if isinstance(callee, MatchFailure):
            return callee
        return dataclasses.replace(
            matched,
            pattern_id="call.function",
            branch_id="function_call",
            captures={**matched.captures, "callee": callee},
            children=tuple(
                AstChild(f"arg_{index}", ExpressionPattern(), argument, "call_argument", "argument")
                for index, argument in enumerate(node.args)
            ),
        )
    schema = callee if isinstance(callee, runtime.OpSchema) else expr.schema_of(callee)
    if not isinstance(schema, runtime.OpSchema):
        return None
    children = schema_children(node, schema, context)
    if children is None or isinstance(children, MatchFailure):
        return children
    return dataclasses.replace(
        matched,
        pattern_id="call.operation",
        branch_id="operation_call",
        captures={**matched.captures, "schema": schema},
        children=children,
    )


class CallPattern(ElementPattern):
    element_name = "op_call"
    syntax = LazyPattern(
        lambda: BindPattern(
            AstNodePattern(
                ast.Call,
                FieldPattern(
                    "func", ChoicePattern(ReferencePattern(), AstNodePattern(ast.Subscript))
                ),
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
            enter_call,
        )
    )

    PARAM_PATTERNS = {
        "variadic": VariadicInputsPattern,
        "tensor": TensorPattern,
        "dtype": DTypePattern,
        "storage": StoragePattern,
        "layout": LayoutPattern,
        "op": OpValuePattern,
        "static": StaticValuePattern,
    }
    ATTRIBUTE_PATTERNS = tuple(PARAM_PATTERNS.values())

    def child_values(self, match, child, children, context):
        if match.branch_id != "operation_call" or child.role != "layout":
            return {}
        storage = next((c for c in match.children if c.name == "attr_storage"), None)
        if storage is not None:
            value = parse_node(
                storage.pattern,
                storage.node,
                context.child(situation=storage.situation, role=storage.role),
            )
        else:
            value = None
        source = next((v for k, v in children.items() if k.startswith("input_")), None)
        return {
            "layout_storage": value
            or (source.type.storage if source is not None else runtime.StorageKind.GMEM)
        }

    def construct(self, match, children, context):
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
                name.removeprefix("attr_"): (
                    value.layout if isinstance(value, PlacedLayout) else value
                )
                for name, value in children.items()
                if name.startswith("attr_")
            }
            if schema.op_class in {runtime.Reshard, runtime.TensorView, runtime.Bitcast}:
                for name, value in tuple(attrs.items()):
                    if isinstance(value, runtime.LayoutBase):
                        attrs[name] = types.normalize_layout(value)
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
            return runtime.Call(
                type=runtime.TensorType.from_scalar(runtime.DType.f32),
                target=operation,
                args=inputs,
            )
        elif match.branch_id == "function_ref_call":
            reference, operands, _ = build_slot_reference(match, children, context)
            args = tuple(value for name, value in children.items() if name.startswith("arg_"))
            return runtime.Call(type=runtime.UnitType(), target=reference, args=(*operands, *args))
        elif match.branch_id == "function_call":
            callee = match.captures["callee"]
            return runtime.Call(
                type=runtime.UnitType() if context.function.dialect == "tir" else callee.return_type,
                target=runtime.FunctionRef((), callee.name),
                args=tuple(children.values()),
            )
        raise RuntimeError(f"no constructor branch for {match.branch_id!r}")

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        CallVariadicInputFormRule(),
        CallBindingRule(),
        AuthoredCallScopeRule(),
        CallTypeInferenceRule(),
    )


EXPR_BINARY_KINDS: Mapping[type[ast.AST], str] = MappingProxyType(
    {
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
)


def expression_operator_pattern(operator_type: type[ast.AST]) -> ChoicePattern:
    return ChoicePattern(
        *(
            AstNodePattern(node_type)
            for node_type in EXPR_BINARY_KINDS
            if issubclass(node_type, operator_type)
        )
    )


CALL_RESULT_RULES: tuple[AstRule[Any], ...] = (
    CallBindingRule(),
    AuthoredCallScopeRule(),
    CallTypeInferenceRule(),
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

    def construct(self, match, children, context):
        args = (children["left"], children["right"])
        placeholder_type = next(
            (arg.type for arg in args if getattr(arg, "type", None) is not None),
            runtime.TensorType.from_scalar(runtime.DType.f32),
        )
        return runtime.Call(
            type=placeholder_type,
            target=nn.MatMul(),
            args=args,
        )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = CALL_RESULT_RULES


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
                        expression_operator_pattern(ast.operator),
                    ),
                    CapturePattern(
                        "kind",
                        lambda node, context: EXPR_BINARY_KINDS[type(node.op)],
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
                        SequencePattern(expression_operator_pattern(ast.cmpop)),
                    ),
                    CapturePattern(
                        "kind",
                        lambda node, context: EXPR_BINARY_KINDS[type(node.ops[0])],
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
                        expression_operator_pattern(ast.boolop),
                    ),
                    CapturePattern(
                        "kind",
                        lambda node, context: EXPR_BINARY_KINDS[type(node.op)],
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

    def construct(self, match, children, context):
        args = (children["left"], children["right"])
        placeholder_type = next(
            (arg.type for arg in args if getattr(arg, "type", None) is not None),
            runtime.TensorType.from_scalar(runtime.DType.f32),
        )
        return runtime.Call(
            type=placeholder_type,
            target=runtime.Binary(kind=runtime.BinaryKind[match.captures["kind"]]),
            args=args,
        )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        CallBindingRule(),
        AuthoredCallScopeRule(),
        CallTypeInferenceRule(),
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

    def construct(self, match, children, context):
        operand = children["operand"]
        target = runtime.Unary(kind=runtime.UnaryKind[match.captures["kind"]])
        return runtime.Call(
            type=operand.type,
            target=target,
            args=(operand,),
        )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        CallBindingRule(),
        AuthoredCallScopeRule(),
        CallTypeInferenceRule(),
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
                            type(op) in {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod}
                        ),
                    ),
                ),
                CapturePattern("operator", lambda node, context: BINARY_OPERATORS[type(node.op)]),
                CapturePattern(
                    "dimension_operator",
                    lambda node, context: {
                        ast.Add: "add",
                        ast.Sub: "sub",
                        ast.Mult: "mul",
                        ast.Div: None,
                        ast.FloorDiv: "floordiv",
                        ast.Mod: "mod",
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

    def construct(self, match, children, context):
        left = children["left"]
        right = children["right"]
        op = type(match.node.op)
        if op is ast.Mult and isinstance(right, slice) and not isinstance(left, slice):
            left, right = right, left
        if isinstance(left, slice) or isinstance(right, slice):
            if isinstance(right, slice) or op not in {ast.Add, ast.Sub, ast.Mult}:
                raise ParseError.from_node(
                    match.node, context, "tile windows only support ± c and * c"
                )
            try:
                if op is ast.Mult:
                    start = runtime.simplify_dim("mul", (left.start, right))
                    stop = runtime.simplify_dim("mul", (left.stop, right))
                    step = runtime.simplify_dim("mul", (left.step, right))
                else:
                    offset = right if op is ast.Add else runtime.simplify_dim("mul", (-1, right))
                    start = runtime.simplify_dim("add", (left.start, offset))
                    stop = runtime.simplify_dim("add", (left.stop, offset))
                    step = left.step
            except (TypeError, ValueError, ZeroDivisionError) as error:
                raise ParseError.from_node(match.node, context, str(error)) from error
            return slice(start, stop, runtime.normalize_dim(step))
        if op is ast.Div:
            raise ParseError.from_node(match.node, context, "index arithmetic uses //, not /")
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

    def construct(self, match, children, context):
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

    def construct(self, match, children, context):
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

    def construct(self, match, children, context):
        note_authored_call(match.node, context)
        value = children["value"]
        index = children["index"]
        if not isinstance(value, runtime.Expr):
            raise ParseError.from_node(
                match.node, context, "subscript requires TensorType or TupleType"
            )
        if isinstance(value.type, runtime.TupleType):
            if isinstance(index, bool) or not isinstance(index, (int, runtime.Expr)):
                raise ParseError.from_node(
                    match.node, context, "Tuple subscript requires an integer expression"
                )
            if isinstance(index, int):
                normalized = index + len(value.type.fields) if index < 0 else index
                index = constant(normalized)
            return infer_call(runtime.TupleGetItem(), (value, index), context)
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
                sizes.append(slice_size(begin, end, stride, context, match.node))
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
        sliced = infer_call(
            runtime.Slice(sizes=tuple(sizes), strides=tuple(strides)),
            (value, starts_expr),
            context,
        )
        if not collapsed:
            return sliced
        new_shape = tuple(
            extent for axis, extent in enumerate(sliced.type.shape) if axis not in collapsed
        )
        return infer_call(runtime.Reshape(new_shape=new_shape), (sliced,), context)

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


def enter_mesh_coordinate(
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
        if candidate < len(types.flatten(mesh.layout).shape):
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


class MeshCoordinatePattern(ElementPattern):
    element_name = "mesh_coordinate"
    syntax = LazyPattern(
        lambda: BindPattern(
            AstNodePattern(
                ast.Attribute,
                FieldPattern("value", AstNodePattern(ast.Name)),
                FieldPattern("attr", LiteralPattern(value_type=str)),
            ),
            enter_mesh_coordinate,
        )
    )

    def construct(self, match, children, context):
        if context.function is None:
            raise ParseError.from_node(match.node, context, "mesh coordinate lacks context")
        note_authored_call(match.node, context)
        mesh = match.captures["mesh"]
        axis = match.captures["axis"]
        extent = types.flatten(mesh.layout).shape[axis]
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise ParseError.from_node(
                match.node, context, "mesh coordinate requires a concrete axis extent"
            )
        cache_key = (id(mesh), axis)
        cached = context.function.state.mesh_coordinates.get(cache_key)
        if cached is not None:
            return cached
        index = constant(axis)
        coordinate = infer_call(runtime.MeshCoord(mesh=mesh), (index,), context)
        coordinate.attach_metadata(expr.RangeMetadata(0, extent))
        context.function.state.cache_mesh_coordinate(cache_key, coordinate)
        return coordinate

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


class ExpressionPattern(ElementPattern):
    element_name = "runtime_expression"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            CeildivExpressionPattern(),
            CallPattern(),
            LaunchPattern(),
            SubscriptExpressionPattern(),
            MatMulExpressionPattern(),
            BinaryExpressionPattern(),
            ConstantPattern(),
            UnaryExpressionPattern(),
            MeshCoordinatePattern(),
            NamePattern(),
            TupleExpressionPattern(),
            TensorPattern(),
            BranchPattern(
                "attribute_expr",
                AstNodePattern(ast.Attribute),
                pattern_id="expression.attribute",
            ),
        )
    )

    def construct(self, match, children, context):
        value = resolve_reference(match.node, context)
        if isinstance(value, runtime.Expr):
            return value
        if isinstance(value, (bool, int, float)):
            return constant(value)
        raise ParseError.from_node(match.node, context, "attribute did not resolve to Expr")

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


@dataclass(frozen=True)
class MeshSelection:
    selection: types.Mesh
    layout: types.Layout
    names: tuple[str, ...]
    outer: types.Layout | None = None


def finish_mesh_context(mesh, match, context):
    binding = context.values.get("mesh_binding")
    enter_mesh_scope(context, mesh, match)
    if isinstance(binding, str):
        context.lexical_scope.define_mesh(binding, mesh)
    context.function.state.mesh_stack.append(mesh)
    return mesh


@dataclass(frozen=True)
class SelectionLevelRule:
    STATEMENT: ClassVar[str] = "A mesh selection names exactly one topology level."

    def apply(self, value, *, match, context):
        if isinstance(value, MeshSelection) and len(value.selection.topologies) != 1:
            raise ParseError.from_node(
                match.node, context, "a mesh selection selects one topology level"
            )
        return value


@dataclass(frozen=True)
class SelectionSizeRule:
    STATEMENT: ClassVar[str] = (
        "A mesh selection layout has positive static extents, static strides, "
        "matching shape/stride and name ranks, and the same size as its selection."
    )

    def apply(self, value, *, match, context):
        if not isinstance(value, MeshSelection):
            return value
        source = types.flatten(types.levels(value.selection)[0])
        layout = value.layout
        message = None
        if types.size(layout) != types.size(source):
            message = (
                "mesh selection layout must hold the same number of positions as its selection"
            )
        elif not all(isinstance(one, int) and one > 0 for one in (*source.shape, *layout.shape)):
            message = "a mesh selection requires positive static extents"
        elif value.names and len(value.names) != len(layout.shape):
            message = "mesh selection names must have the rank of its shape"
        elif len(layout.strides) != len(layout.shape):
            message = "mesh selection strides must have the rank of its shape"
        elif not all(isinstance(step, int) for step in (*source.strides, *layout.strides)):
            message = "a mesh selection requires static strides"
        if message is not None:
            raise ParseError.from_node(match.node, context, message)
        return value


@dataclass(frozen=True)
class SelectionCoverRule:
    STATEMENT: ClassVar[str] = "A mesh selection layout covers each selection-local index exactly once."

    def apply(self, value, *, match, context):
        if not isinstance(value, MeshSelection):
            return value
        layout = value.layout
        order = types.compact_row_major(layout.shape)
        indices = {
            types.crd2idx(types.idx2crd(index, layout.shape, order), layout.shape, layout.strides)
            for index in range(types.size(layout))
        }
        if indices != set(range(types.size(layout))):
            raise ParseError.from_node(
                match.node, context, "mesh selection layout must cover each selected position once"
            )
        return value


@dataclass(frozen=True)
class SelectionStrideRule:
    """Composition uses AssertionError to reject indivisible mode strides."""

    STATEMENT: ClassVar[str] = "Each new mesh selection axis maps to one fixed physical stride."

    def apply(self, value, *, match, context):
        if not isinstance(value, MeshSelection):
            return value
        source = types.flatten(types.levels(value.selection)[0])
        try:
            outer = layout_algebra.composition(source, value.layout, major="row")
        except (AssertionError, ValueError, TypeError) as error:
            raise ParseError.from_node(
                match.node,
                context,
                "a mesh selection axis does not map to one stride of the selection",
            ) from error
        if outer.shape != value.layout.shape:
            raise ParseError.from_node(
                match.node,
                context,
                "a mesh selection axis does not map to one stride of the selection",
            )
        return dataclasses.replace(value, outer=outer)


@dataclass(frozen=True)
class SelectionConstructionRule:
    STATEMENT: ClassVar[str] = (
        "A checked selection constructs a Mesh with its topology, sliced at its offset "
        "unless it holds its whole level, enters its execution scope, and defines its "
        "lexical binding."
    )

    def apply(self, value, *, match, context):
        if not isinstance(value, MeshSelection):
            return value
        assert value.outer is not None
        start = types.starts(value.selection)[0]
        whole = start == 0 and types.size(value.outer) == value.selection.topologies[0].size
        mesh = runtime.Mesh(
            value.selection.topologies,
            value.outer if whole else types.ComposedLayout(None, start, value.outer),
            value.names,
        )
        return finish_mesh_context(mesh, match, context)


@dataclass(frozen=True)
class SelectionBindingRule:
    STATEMENT: ClassVar[str] = "A mesh selection names a lexical Mesh binding or its constant slice."

    def apply(self, value, *, node, context):
        reference = node.value if isinstance(node, ast.Subscript) else node
        if not isinstance(reference, ast.Name) or context.lexical_scope.lookup_mesh(reference.id) is None:
            return PatternFailure(
                "mesh", node, f"{ast.unparse(reference)!r} is not a lexical Mesh binding"
            )
        return value


@dataclass(frozen=True)
class SelectionLayoutFormRule:
    STATEMENT: ClassVar[str] = "A mesh selection layout is a shape tuple or a shape/stride tuple pair."

    def apply(self, value, *, node, context):
        shape_context = context.child(situation="mesh_shape", role="shape")
        if isinstance(node, ast.Tuple) and ShapePattern().match(node, shape_context) is not None:
            return node, None
        if (
            not isinstance(node, ast.Tuple) or len(node.elts) != 2
            or any(not isinstance(one, ast.Tuple) or ShapePattern().match(one, shape_context) is None
                   for one in node.elts)
        ):
            return PatternFailure(
                "mesh", node, "a mesh selection layout is a shape tuple or (shape, strides) tuple"
            )
        return tuple(node.elts)


def enter_mesh_context(
    node: object, context: MatchContext, matched: AstMatch[Any]
) -> AstMatch[Any] | MatchFailure | None:
    assert isinstance(node, ast.Call)
    if not node.args:
        return None
    selecting = not isinstance(node.args[0], ast.Tuple)
    if selecting:
        checked = MeshContextPattern.BIND_RULES[0].apply(
            matched, node=node.args[0], context=context
        )
        if isinstance(checked, MatchFailure):
            return checked
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
            return PatternFailure("mesh", node, "`names` is already bound by a positional argument")
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
    shape_node, strides_node = layout_node, None
    if selecting:
        checked = MeshContextPattern.BIND_RULES[1].apply(matched, node=layout_node, context=context)
        if isinstance(checked, MatchFailure):
            return checked
        shape_node, strides_node = checked
    children = [
        AstChild(
            "selection" if selecting else "topology_names",
            StaticValuePattern(),
            node.args[0],
            "mesh_topologies",
            "topologies",
        ),
        AstChild(
            "shape" if selecting else "layout",
            ShapePattern() if selecting else LayoutPattern(),
            shape_node if selecting else layout_node,
            "mesh_shape" if selecting else "mesh_layout",
            "shape" if selecting else "layout",
        ),
    ]
    if strides_node is not None:
        children.append(
            AstChild("strides", ShapePattern(), strides_node, "mesh_strides", "strides")
        )
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
        branch_id="mesh_selection" if selecting else "mesh_context",
        children=tuple(children),
    )


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
                enter_mesh_context,
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

    def construct(self, match, children, context):
        if context.function is None:
            raise ParseError.from_node(match.node, context, "Mesh requires function context")
        if match.branch_id == "mesh_selection":
            shape, strides = children["shape"], children.get("strides")
            layout = types.flatten(
                runtime.Layout(
                    shape, types.compact_row_major(shape) if strides is None else strides
                )
            )
            return MeshSelection(children["selection"], layout, children.get("names", ()))
        elif match.branch_id == "mesh_context":
            topology_names = children["topology_names"]
            if not isinstance(topology_names, tuple):
                raise ParseError.from_node(
                    match.node,
                    context,
                    "Mesh topologies must be a tuple",
                )
            names = children.get("names", ())
            if any(isinstance(topology, str) for topology in topology_names):
                topology_names = resolve_mesh_topologies_at(
                    topology_names, context.function.topologies, match.node, context
                )
            layout = children["layout"]
            if isinstance(layout, PlacedLayout):
                layout = layout.layout
            try:
                mesh = runtime.Mesh(
                    topologies=topology_names,
                    layout=layout,
                    names=names,
                )
            except (TypeError, ValueError) as error:
                raise ParseError.from_node(match.node, context, str(error)) from error
        elif match.branch_id == "mesh_reference":
            mesh = children["value"]
            if not isinstance(mesh, runtime.Mesh):
                raise ParseError.from_node(match.node, context, "with context is not Mesh")
        else:
            raise RuntimeError(f"no constructor branch for {match.branch_id!r}")
        return finish_mesh_context(mesh, match, context)

    BIND_RULES = (SelectionBindingRule(), SelectionLayoutFormRule())
    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        SelectionLevelRule(), SelectionSizeRule(), SelectionCoverRule(), SelectionStrideRule(),
        SelectionConstructionRule(),
    )


def authored_infer_context(context):
    """The one inference context every node of this function is built against."""
    found = context.lexical_scope.lookup(TYPE_INFER_CONTEXT)
    assert isinstance(found, AuthoredInferContext)
    return found


def reject_unscoped_call(node, context):
    raise ParseError.from_node(
        node,
        context,
        f"{ast.unparse(node)} runs outside every mesh scope in a HIR function "
        "that opens one; move this operation inside with Mesh(...)",
    )


def note_authored_call(node, context):
    """Record authored Calls in their execution scope.

    See [parser §2.2](docs/spec/parser.md#22-rules).
    """
    if context.function is None or context.function.dialect != "hir":
        return
    if authored_infer_context(context).current_mesh is not None:
        return
    state = context.function.state
    if state.opened_mesh:
        reject_unscoped_call(node, context)
    if state.unscoped_call is None:
        state.note_unscoped_call(node)


def directly_bound_names(statements):
    """Return names assigned directly by a block's statements."""
    names: set[str] = set()
    for statement in statements:
        if isinstance(statement, (ast.With, ast.For, ast.AsyncFor, ast.FunctionDef, ast.ClassDef)):
            continue
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
            targets = (statement.target,)
        else:
            continue
        for target in targets:
            names.update(
                name.id
                for name in ast.walk(target)
                if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)
            )
    return frozenset(names)


def loaded_names(statements):
    """Return names read in a sequence of following statements."""
    return frozenset(
        name.id
        for statement in statements
        for name in ast.walk(statement)
        if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Load)
    )


def read_before_bound(statements):
    """Return names a block reads before it binds them, in statement order.

    A name a block reads on its way to binding it came from outside the block:
    an accumulator reads what the last round left before it writes this one.
    """
    bound: set[str] = set()
    live: set[str] = set()
    for statement in statements:
        live.update(loaded_names((statement,)) - bound)
        bound.update(directly_bound_names((statement,)))
    return frozenset(live)


def live_out(statements, after: frozenset[str] = frozenset()) -> dict[ast.With, frozenset[str]]:
    """Each with's conservative live-out: names read by following statements."""
    found: dict[ast.With, frozenset[str]] = {}
    for statement in reversed(statements):
        if isinstance(statement, ast.With):
            found[statement] = after
            found.update(live_out(statement.body, after))
        elif isinstance(statement, (ast.For, ast.While)):
            found.update(live_out(statement.body))
            found.update(live_out(statement.orelse))
        elif isinstance(statement, ast.If):
            found.update(live_out(statement.body, after))
            found.update(live_out(statement.orelse, after))
        after |= loaded_names((statement,))
    return found


def enter_mesh_scope(context, mesh, match):
    """Record the scope a `with` opens, before its body is built.

    Inference runs as each node is constructed, so the region node is not there
    to be walked into when the body is typed -- the `with` says what is in force
    on the way in and puts it back on the way out. The names already bound go
    with it, so on the way out the new ones can be told from the old.
    """
    infer = authored_infer_context(context)
    if context.function.dialect == "hir":
        state = context.function.state
        if state.unscoped_call is not None:
            reject_unscoped_call(state.unscoped_call, context)
        state.open_mesh()
    try:
        entered_mesh = runtime.make_mesh(infer.current_mesh, mesh) if infer.current_mesh else mesh
    except ValueError as error:
        raise ParseError.from_node(match.node, context, str(error)) from error
    context.lexical_scope.push_frame()
    context.lexical_scope.define(
        TYPE_INFER_CONTEXT,
        dataclasses.replace(infer, current_mesh=entered_mesh),
    )


def scoped_region(mesh, body, params=(), args=()):
    """Build a type-transparent HIR mesh region."""
    return runtime.MeshRegion(
        mesh=mesh,
        params=tuple(params),
        args=tuple(args),
        body=body,
        type=body.type,
    )


def region_captures(node, context, excluded=frozenset()):
    """Capture outer expression bindings for one region boundary."""

    def free_names(statements, outer_bound=frozenset()):
        local = directly_bound_names(statements) - read_before_bound(statements)
        visible = outer_bound | local
        found = set()

        def visit(statement, bound):
            if isinstance(statement, ast.With):
                for item in statement.items:
                    found.update(
                        name.id
                        for name in ast.walk(item.context_expr)
                        if (
                            isinstance(name, ast.Name)
                            and isinstance(name.ctx, ast.Load)
                            and name.id not in bound
                        )
                    )
                nested_bound = set(bound)
                for item in statement.items:
                    if isinstance(item.optional_vars, ast.Name):
                        nested_bound.add(item.optional_vars.id)
                nested_bound.update(
                    directly_bound_names(statement.body) - read_before_bound(statement.body)
                )
                for child in statement.body:
                    visit(child, frozenset(nested_bound))
                return
            if isinstance(statement, (ast.For, ast.AsyncFor)):
                found.update(
                    name.id
                    for name in ast.walk(statement.iter)
                    if isinstance(name, ast.Name)
                    and isinstance(name.ctx, ast.Load)
                    and name.id not in bound
                )
                targets = {
                    name.id for name in ast.walk(statement.target) if isinstance(name, ast.Name)
                }
                found.update(free_names(statement.body, bound | targets))
                found.update(free_names(statement.orelse, bound))
                return
            for name in ast.walk(statement):
                if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Load):
                    if name.id not in bound:
                        found.add(name.id)

        for statement in statements:
            visit(statement, visible)
        return found

    free = free_names(node.body, excluded)
    params = []
    args = []
    for name in sorted(free):
        value = context.lexical_scope.lookup(name)
        if isinstance(value, (bool, int, float)):
            value = constant(value)
        if isinstance(value, slice):
            value = value.start
        if not isinstance(value, runtime.Expr):
            continue
        params.append(runtime.Var(type=value.type, name=name))
        args.append(value)
    return tuple(params), tuple(args)


def region_capture_bindings(params, context):
    """Rebuild captured tile windows around the region's new induction parameter."""
    bindings = {}
    for param in params:
        value = context.lexical_scope.lookup(param.name)
        if isinstance(value, slice):
            width = runtime.normalize_dim(runtime.simplify_dim("sub", (value.stop, value.start)))
            stop = runtime.simplify_dim("add", (param, width))
            bindings[param.name] = slice(param, stop, value.step)
        else:
            bindings[param.name] = param
    return bindings


def bind_region_results(context, region, names, node):
    """Bind one region directly or project each result from its tuple."""
    if len(names) == 1:
        context.lexical_scope.define(names[0], region)
        return
    for index, name in enumerate(names):
        projection = infer_call(runtime.TupleGetItem(), (region, constant(index)), context)
        attach_authored_metadata(
            projection,
            node,
            dataclasses.replace(context, binding_name=name),
        )
        context.lexical_scope.define(name, projection)


def rebind_through_region(context, mesh, names, frame, node, params=(), args=()):
    """Point escaped names at a region result, whether or not it has a body value.

    A region's computed values are named, and those names are how code after it
    reads them. Rebinding each through the region is what keeps who-ran-this on
    the graph instead of leaving it to be guessed from a layout. The first name
    rebound comes back, so a caller with no body value of its own still has one
    that reaches the region.
    """
    values = [(name, frame[name]) for name in names]
    if len(values) == 1:
        scoped = scoped_region(mesh, values[0][1], params, args)
        bind_region_results(context, scoped, [values[0][0]], node)
        return values[0][0]
    tuple_type = runtime.TupleType(fields=tuple(value.type for _, value in values))
    tuple_body = runtime.IrTuple(type=tuple_type, elements=tuple(value for _, value in values))
    scoped = scoped_region(mesh, tuple_body, params, args)
    bind_region_results(context, scoped, [name for name, _ in values], node)
    return values[0][0]


def enter_with(node: object, context: MatchContext, matched: AstMatch[Any]) -> AstMatch[Any] | None:
    assert isinstance(node, ast.With)
    item = node.items[0]
    assert isinstance(item.optional_vars, ast.Name)
    binding = item.optional_vars.id
    params, args = region_captures(node, context)
    return dataclasses.replace(
        matched,
        pattern_id="statement.with_mesh",
        branch_id="with_mesh",
        captures={
            **matched.captures,
            "binding": binding,
            "region_params": params,
            "region_args": args,
        },
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
                body_as_ast_module(node.body),
                "block",
                "with_body",
                lexical_bindings=(
                    region_capture_bindings(params, context)
                    if context.function.dialect == "hir"
                    else None
                ),
                values={
                    "loop_captures": (*context.values.get("loop_captures", ()), *zip(params, args))
                },
            ),
        ),
    )


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
                        transform=body_as_ast_module,
                    ),
                ),
            ),
            enter_with,
        )
    )

    def construct(self, match, children, context):
        if context.function is None:
            raise ParseError.from_node(match.node, context, "`with` requires function context")
        mesh = children["mesh"]
        frame = context.lexical_scope.pop_frame()
        frame.pop(match.captures["binding"], None)
        if not context.function.state.mesh_stack:
            raise ParseError.from_node(match.node, context, "Mesh stack is unbalanced")
        active_mesh = context.function.state.mesh_stack.pop()
        if active_mesh is not mesh:
            raise ParseError.from_node(match.node, context, "Mesh stack is unbalanced")
        if context.function.dialect == "hir":
            body = children["body"]
            params = match.captures.get("region_params", ())
            entry = {param.name: param for param in params}
            live = context.live_out[match.node]
            escaping = tuple(
                name
                for name, value in frame.items()
                if isinstance(value, runtime.Expr)
                and value is not entry.get(name)
                and (context.lexical_scope.lookup(name) is not None or name in live)
            )
            args = match.captures.get("region_args", ())
            rebound = None
            if escaping:
                rebound = rebind_through_region(
                    context, mesh, escaping, frame, match.node, params, args
                )
            if body is not None and (
                rebound is None or isinstance(match.node.body[-1], ast.Return)
            ):
                return scoped_region(mesh, body, params, args)
            return None if rebound is None else context.lexical_scope.lookup(rebound)
        binding = runtime.Var(
            type=runtime.TensorType.from_scalar(
                runtime.DType.i64, storage=runtime.StorageKind.RMEM
            ),
            name=match.captures["binding"],
        )
        return runtime.MeshScope(mesh=mesh, binding=binding, body=children["body"])

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


def enter_launch(node: object, context: MatchContext, matched: AstMatch[Any]):
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
    reference = slot_reference(node.args[0], context)
    children = [
        *(
            reference_index_children(reference[1])
            if reference
            else (AstChild("callee", StaticValuePattern(), node.args[0], "launch_callee"),)
        ),
        *(
            AstChild(f"arg_{index}", ExpressionPattern(), argument, "launch_argument")
            for index, argument in enumerate(node.args[1:])
        ),
        AstChild("grid", StaticValuePattern(), keywords["grid"], "launch_extent"),
        AstChild("block", StaticValuePattern(), keywords["block"], "launch_extent"),
    ]
    for name in ("cluster", "dynamic_smem", "stream", "attrs"):
        if name in keywords:
            children.append(AstChild(name, StaticValuePattern(), keywords[name], "launch_option"))
    return dataclasses.replace(
        matched,
        pattern_id="statement.launch",
        branch_id="launch",
        captures={"arg_count": len(node.args) - 1, "reference": reference},
        children=tuple(children),
    )


class LaunchPattern(ElementPattern):
    """TIR host launch statement lowered from the authored ``launch`` call."""

    element_name = "launch"
    syntax = LazyPattern(lambda: BindPattern(AstNodePattern(ast.Call), enter_launch))

    def construct(self, match, children, context):
        if match.captures["reference"]:
            ref, _, callee = build_slot_reference(match, children, context, launch=True)
        else:
            callee = children["callee"]
            if isinstance(callee, runtime.Module):
                raise ParseError.from_node(
                    match.node.args[0], context, "a launched Module must be a slot of this @module"
                )
            callee = declared_prototype(callee, match.node.args[0], context)
            if isinstance(callee, MatchFailure):
                raise ParseError.from_node(match.node.args[0], context, callee.detail)
            ref = runtime.FunctionRef((), callee.name)
        options = {
            name: children[name]
            for name in ("cluster", "dynamic_smem", "stream", "attrs")
            if name in children
        }
        return tir.launch.launch_call(
            callee,
            tuple(children[f"arg_{index}"] for index in range(match.captures["arg_count"])),
            children["grid"],
            children["block"],
            ref=ref,
            **options,
        )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


def carry_statement_names(node: object, context: MatchContext) -> tuple[str, ...]:
    """Names a statement binds that the loop carries rather than the loop itself.

    An assignment states its own; a `with` states what its body binds,
    because a region delimits who runs the work and not who owns the value.
    Every other statement binds nothing the loop carries.
    """
    if not isinstance(node, ast.With):
        return ()
    found: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        target = child.targets[0]
        targets = target.elts if isinstance(target, ast.Tuple) else (target,)
        found.extend(
            item.id for item in targets if isinstance(item, ast.Name) and item.id not in found
        )
    return tuple(found)


def carry_target_names(node: object, context: MatchContext) -> tuple[str, ...]:
    assert isinstance(node, ast.Assign)
    target = node.targets[0]
    targets = target.elts if isinstance(target, ast.Tuple) else (target,)
    return tuple(item.id for item in targets if isinstance(item, ast.Name))


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
                    CapturePattern("names", carry_target_names),
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
                            transform=body_as_ast_module,
                        ),
                    ),
                ),
                pattern_id="loop.carry_statement",
            ),
            BranchPattern(
                "loop_carry_statement",
                AstNodePattern(
                    ast.stmt,
                    CapturePattern("names", carry_statement_names),
                ),
                pattern_id="loop.carry_statement",
            ),
        )
    )

    def construct(self, match, children, context):
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

    def construct(self, match, children, context):
        names: list[str] = []
        for statement_names in children.values():
            for name in statement_names:
                if name not in names:
                    names.append(name)
        return tuple(names)

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


def tile_iterator_callee():
    return AstNodePattern(
        ast.Attribute,
        FieldPattern("value", AstNodePattern(ast.Name, FieldPattern("id", LiteralPattern("tf")))),
        FieldPattern("attr", LiteralPattern("tile")),
    )


@dataclass(frozen=True)
class TileIteratorQualificationRule:
    STATEMENT: ClassVar[str] = "A tiled loop uses tf.tile rather than a bare tile call."

    def apply(self, value, *, node, context):
        if loop_iterator_kind(node) == "bare_tile":
            return PatternFailure(
                "loop_header", node.func, "tile loops are written tf.tile(extent, step)"
            )
        return value


def loop_iterator_kind(call: ast.Call) -> str | None:
    """Recognize the iterator surface shared by syntax and diagnostics."""
    callee = call.func
    if isinstance(callee, ast.Name) and callee.id in {"tile", "range"}:
        return "bare_tile" if callee.id == "tile" else "range"
    if tile_iterator_callee().match(callee, None) is not None:
        return "tile"
    return None


class LoopIteratorPattern(ElementPattern):
    element_name = "loop_iterator"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            BranchPattern(
                "tile",
                AstNodePattern(
                    ast.Call,
                    FieldPattern("func", tile_iterator_callee()),
                    FieldPattern("keywords", SequencePattern()),
                    FieldPattern(
                        "args",
                        ChoicePattern(
                            SequencePattern(AstNodePattern(ast.expr), AstNodePattern(ast.expr)),
                            SequencePattern(
                                AstNodePattern(ast.expr),
                                AstNodePattern(ast.expr),
                                AstNodePattern(ast.expr),
                            ),
                        ),
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

    def construct(self, match, children, context):
        return loop_iterator_kind(match.node)

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


def reads_program_value(node: ast.AST, context: MatchContext) -> bool:
    """Whether a bound reads a mesh coordinate or an expression in lexical scope."""
    return any(
        isinstance(context.lexical_scope.lookup(name.id), (runtime.Mesh, runtime.Expr))
        for name in ast.walk(node)
        if isinstance(name, ast.Name)
    )


def iterator_arity_failure(kind: str, count: int, node: ast.Call) -> PatternFailure | None:
    """Describe an invalid tile/range arity, if this call has one."""
    if count in ({2, 3} if kind == "tile" else {1, 2, 3}):
        return None
    if kind == "tile" and count == 1:
        detail = "tf.tile(extent) is not supported; use range(extent)"
    elif kind == "tile":
        detail = (
            f"tf.tile() takes 2 or 3 arguments, (stop, step) or (start, stop, step), got {count}"
        )
    else:
        detail = f"range() takes 1 to 3 arguments, got {count}"
    return PatternFailure("loop_header", node, detail)


def bound_loop_pattern(node: ast.AST, context: MatchContext) -> AstPattern[Any]:
    if context.function is not None and context.function.dialect == "tir":
        return ChoicePattern(ExpressionPattern(), StaticValuePattern())
    if reads_program_value(node, context):
        return ExpressionPattern()
    return StaticValuePattern()


def enter_loop_header(
    node: object, context: MatchContext, matched: AstMatch[Any]
) -> AstMatch[Any] | MatchFailure | None:
    assert isinstance(node, ast.For)
    assert isinstance(node.target, ast.Name)
    assert isinstance(node.iter, ast.Call)
    kind = loop_iterator_kind(node.iter)
    count = len(node.iter.args)
    if node.iter.keywords:
        return PatternFailure(
            "loop_header",
            node.iter,
            "tf.tile()/range() does not accept keyword args (positional-only at the IR level)",
        )
    if failure := iterator_arity_failure(kind, count, node.iter):
        return failure
    if kind == "tile" and count == 2:
        fields = ("extent", "step")
        defaults = {"start": 0}
    elif kind == "tile":
        fields = ("start", "extent", "step")
        defaults = {}
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
            body_as_ast_module(node.body),
            "loop_carry",
        )
    ]
    children.extend(
        AstChild(
            field_name,
            bound_loop_pattern(argument, context),
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
                        transform=body_as_ast_module,
                    ),
                ),
            ),
            enter_loop_header,
        )
    )

    def match(self, node: object, context: MatchContext) -> AstMatch[Any] | MatchFailure | None:
        """Name the invalid iterator before the shape-exact syntax rejects it.

        The syntax encodes each iterator's arity so the generated grammar shows
        what the parser accepts. A shape mismatch alone would report only that
        the pattern did not match, so the specific reason is stated here first.
        """
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            kind = loop_iterator_kind(node.iter)
            count = len(node.iter.args)
            for rule in LoopHeaderPattern.BIND_RULES:
                checked = rule.apply(None, node=node.iter, context=context)
                if isinstance(checked, MatchFailure):
                    return checked
            if kind not in {"tile", "range"}:
                return PatternFailure(
                    "loop_header",
                    node.iter.func,
                    "loop iterator must be tf.tile(...) or range(...)",
                )
            if node.iter.keywords:
                return PatternFailure(
                    "loop_header",
                    node.iter,
                    "tf.tile()/range() does not accept keyword args (positional-only at the IR level)",
                )
            if failure := iterator_arity_failure(kind, count, node.iter):
                return failure
        return super().match(node, context)

    def construct(self, match, children, context):
        if context.function is not None and context.function.dialect == "tir":
            iv = runtime.Var(
                type=runtime.TensorType.from_scalar(runtime.DType.i64),
                name=match.captures["target"],
            )
            values = dict(match.captures["defaults"])
            values.update({name: value for name, value in children.items() if name != "carry"})
            bounds = [values[name] for name in ("start", "extent", "step")]
            bounds = [constant(v) if isinstance(v, (bool, int, float)) else v for v in bounds]
            bounds = [
                bound
                if runtime.static_dim_value(bound) is not None
                else runtime.normalize_dim(bound)
                for bound in bounds
            ]
            context.lexical_scope.push_frame()
            context.lexical_scope.define(match.captures["target"], iv)
            inference = context.lexical_scope.lookup(TYPE_INFER_CONTEXT)
            context.lexical_scope.define(
                TYPE_INFER_CONTEXT,
                dataclasses.replace(inference, loops=(*inference.loops, (iv, *bounds))),
            )
            return (iv, *bounds)
        if context.function is None or context.function.dialect != "hir":
            raise ParseError.from_node(match.node, context, "HIR loop header in a non-HIR context")
        values = dict(match.captures["defaults"])
        values.update((name, value) for name, value in children.items() if name != "carry")
        try:
            start = runtime.normalize_dim(values["start"])
            extent = runtime.normalize_dim(values["extent"])
            step = runtime.normalize_dim(values["step"])
        except (TypeError, ValueError) as error:
            raise ParseError.from_node(match.node, context, str(error)) from error
        induction_var = runtime.Var(
            type=runtime.TensorType.from_scalar(runtime.DType.i64),
            name=match.captures["target"],
        )
        carry_names = tuple(
            name
            for name in children["carry"]
            if isinstance(context.lexical_scope.lookup(name), runtime.Expr)
        )
        carry_inits = tuple(context.lexical_scope.lookup(name) for name in carry_names)
        phi_vars = tuple(
            runtime.Var(type=value.type, name=name) for name, value in zip(carry_names, carry_inits)
        )
        params, args = region_captures(
            match.node, context, frozenset((match.captures["target"], *carry_names))
        )
        bindings = region_capture_bindings(params, context)
        context.lexical_scope.push_frame()
        for name, value in bindings.items():
            context.lexical_scope.define(name, value)
        if match.captures["kind"] == "tile":
            stop = runtime.simplify_dim("add", (induction_var, runtime.dim_expr(step)))
            binding = slice(induction_var, stop, 1)
        else:
            binding = induction_var
        context.lexical_scope.define(match.captures["target"], binding)
        inference = context.lexical_scope.lookup(TYPE_INFER_CONTEXT)
        context.lexical_scope.define(
            TYPE_INFER_CONTEXT,
            dataclasses.replace(
                inference.with_captures(zip(params, args)),
                loops=(*inference.loops, (induction_var, start, extent, step)),
            ),
        )
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
            params=(*phi_vars, *params),
            args=(*carry_inits, *args),
        )

    BIND_RULES = (TileIteratorQualificationRule(),)
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
                            isinstance(statement, (ast.Return, ast.Expr, ast.Pass))
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
        ),
    )

    def construct(self, match, children, context):
        """The loop body's value: the last statement that has one.

        A `with` that only rebinds the names escaping it has no value of its
        own -- the names it rebound reach its region, and the loop reads them
        as its carried values -- so the body's value is the statement before.
        """
        if not children:
            raise ParseError.from_node(match.node, context, "loop body cannot be empty")
        value = next(
            (item for item in reversed(tuple(children.values())) if item is not None), None
        )
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
                        ChoicePattern(
                            ConditionPattern(
                                "tir loop body",
                                lambda node, context: (
                                    context.function is not None
                                    and context.function.dialect == "tir"
                                ),
                                BlockPattern(),
                            ),
                            ConditionPattern(
                                "hir loop body",
                                lambda node, context: (
                                    context.function is None or context.function.dialect == "hir"
                                ),
                                LoopBodyPattern(),
                            ),
                        ),
                        "loop_body",
                        transform=body_as_ast_module,
                    ),
                ),
            ),
            pattern_id="statement.for",
        )
    )

    def construct(self, match, children, context):
        frame = children["header"]
        body = children["body"]
        if context.function is not None and context.function.dialect == "tir":
            context.lexical_scope.pop_frame()
            induction_var, start, stop, step = frame
            return runtime.For(induction_var, start, stop, step, body)
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
        grid = runtime.LoopRegion(
            type=result_type,
            induction_var=frame.induction_var,
            params=frame.params,
            args=frame.args,
            body=body,
            yield_values=yield_values,
            start=frame.start,
            extent=frame.extent,
            step=frame.step,
        )
        bind_region_results(context, grid, frame.carry_names, match.node)
        return grid

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


@dataclass(frozen=True)
class TirOnlyStatementRule:
    STATEMENT: ClassVar[str] = "A TIR-only statement must appear in a prim_func."

    def apply(self, value, *, match, context):
        if context.function is None or context.function.dialect != "tir":
            raise ParseError.from_node(
                match.node,
                context,
                f"{match.element_name} is a TIR statement; HIR does not support it",
            )
        return value


class IfPattern(ElementPattern):
    element_name = "if"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "if",
            AstNodePattern(
                ast.If,
                CapturePattern("cond_node", lambda node, context: node.test),
                FieldPattern(
                    "body",
                    ChildPattern("then", BlockPattern(), "block", transform=body_as_ast_module),
                ),
                FieldPattern(
                    "orelse",
                    OptionalPattern(
                        ChildPattern("else", BlockPattern(), "block", transform=body_as_ast_module)
                    ),
                ),
            ),
            pattern_id="statement.if",
        )
    )

    def construct(self, match, children, context):
        return runtime.If(
            tir_scalar_expr(match.captures["cond_node"], context),
            children["then"],
            children.get("else", runtime.Sequential(body=())),
        )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (TirOnlyStatementRule(),)


class WhilePattern(ElementPattern):
    element_name = "while"
    syntax = LazyPattern(
        lambda: BranchPattern(
            "while",
            AstNodePattern(
                ast.While,
                CapturePattern("cond_node", lambda node, context: node.test),
                FieldPattern(
                    "body",
                    ChildPattern("body", BlockPattern(), "block", transform=body_as_ast_module),
                ),
            ),
            pattern_id="statement.while",
        )
    )

    def construct(self, match, children, context):
        return runtime.While(
            tir_scalar_expr(match.captures["cond_node"], context), children["body"]
        )

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (TirOnlyStatementRule(),)


def tir_scalar_expr(node: ast.expr, context):
    if isinstance(node, ast.Name):
        value = context.lexical_scope.lookup(node.id)
        if isinstance(value, runtime.Expr):
            return value
    if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int)):
        return constant(node.value)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        kind = EXPR_BINARY_KINDS.get(type(node.ops[0]))
        if kind is not None:
            lhs = tir_scalar_expr(node.left, context)
            rhs = tir_scalar_expr(node.comparators[0], context)
            return runtime.Call(
                type=runtime.TensorType.from_scalar(runtime.DType.bool),
                target=runtime.Binary(kind=runtime.BinaryKind[kind]),
                args=(lhs, rhs),
            )
    raise ParseError.from_node(
        node,
        context,
        "TIR if predicate must be an integer/bool constant, scalar variable, or supported comparison",
    )


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
                CapturePattern(
                    "target_nodes",
                    lambda node, context: tuple(node.targets[0].elts),
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

    def construct(self, match, children, context):
        value = children["value"]
        names = match.captures["names"]
        target_nodes = match.captures["target_nodes"]
        if not isinstance(value.type, runtime.TupleType):
            raise ParseError.from_node(
                match.node, context, "tuple assignment requires a TupleType value"
            )
        if len(names) != len(value.type.fields):
            raise ParseError.from_node(
                match.node, context, "tuple assignment arity does not match TupleType"
            )
        schema = expr.schema_of(type(value.target)) if isinstance(value, runtime.Call) else None
        parent_name = getattr(schema, "name", None) or ", ".join(names)
        value.attach_metadata(expr.BindingMetadata(parent_name))
        for index, (name, target_node) in enumerate(zip(names, target_nodes)):
            assert isinstance(target_node, ast.Name)
            projection = infer_call(runtime.TupleGetItem(), (value, constant(index)), context)
            attach_authored_metadata(
                projection,
                target_node,
                dataclasses.replace(context, binding_name=name),
            )
            context.lexical_scope.define(name, projection)
        return value

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


def resolved_mesh(value, match, context):
    """Name the levels a Mesh written in a function body says it divides.

    A Mesh is a value wherever it is written, so one bound in the body has to
    arrive as complete as one a `with` opens: it names its levels by string, and
    the module is what says which levels those are.
    """
    if not isinstance(value, runtime.Mesh) or context.function is None:
        return value
    if not any(isinstance(topology, str) for topology in value.topologies):
        return value
    topologies = resolve_mesh_topologies_at(
        value.topologies,
        context.function.topologies,
        match.node,
        context,
    )
    return dataclasses.replace(value, topologies=topologies)


class StatementPattern(ElementPattern):
    element_name = "statement"
    syntax = LazyPattern(
        lambda: ChoicePattern(
            IfPattern(),
            WhilePattern(),
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

    def construct(self, match, children, context):
        if match.branch_id == "assignment":
            value = children.get("value")
            annotation = children.get("annotation")
            if value is None:
                raise ParseError.from_node(match.node, context, "assignment requires a value")
            value = resolved_mesh(value, match, context)
            if context.function.dialect == "tir" and not isinstance(value, runtime.Expr):
                context.lexical_scope.define(match.captures["name"], value)
                return None
            if isinstance(annotation, clause.WhereMetadata):
                if context.function.dialect != "hir" or not isinstance(
                    value.type, types.TensorType
                ):
                    raise ParseError.from_node(
                        match.node,
                        context,
                        "where annotation requires a tensor-valued HIR Expr",
                    )
                previous = expr.get_metadata(value, clause.WhereMetadata)
                if previous is not None:
                    binding = expr.get_metadata(value, expr.BindingMetadata)
                    label = binding.name if binding is not None else "<unnamed>"
                    raise ParseError.from_node(
                        match.node,
                        context,
                        f"duplicate where annotation for Expr {label!r}",
                    )
                value.attach_metadata(expr.BindingMetadata(match.captures["name"]))
                value.attach_metadata(annotation)
            elif annotation is not None and value.type != annotation:
                raise ParseError.from_node(
                    match.node, context, "annotated assignment type mismatch"
                )
            name = match.captures["name"]
            if context.function.dialect == "hir":
                if (
                    isinstance(value, runtime.Call)
                    and expr.get_metadata(value, expr.BindingMetadata) is None
                ):
                    value.attach_metadata(expr.BindingMetadata(name))
                context.lexical_scope.define(name, value)
                return value
            variable = runtime.Var(type=value.type, name=name)
            context.lexical_scope.define(name, variable)
            return runtime.LetStmt(variable, value)
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
                        ),
                    ),
                ),
            ),
            pattern_id="function.block",
        ),
    )

    def construct(self, match, children, context):
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

        return runtime.Sequential(body=tuple(value for value in values if value is not None))

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()


@dataclass(frozen=True)
class FunctionSignatureRule:
    STATEMENT: ClassVar[str] = "A function must construct an ordered parameter tuple."

    def apply(self, value, *, match, context):
        if not isinstance(getattr(value, "params", None), tuple):
            raise ParseError.from_node(match.node, context, "function params were not constructed")
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
        if (
            context.function.dialect == "tir"
            and kind != "prim_func"
            and context.function.role is not FunctionRole.VARIANT
        ):
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
class FunctionReturnCompatibilityRule:
    STATEMENT: ClassVar[str] = (
        "A HIR body with a return annotation must satisfy that annotation; a "
        "dispatch prototype must declare one, and each variant body must "
        "satisfy the prototype return contract."
    )

    def apply(self, value, *, match, context):
        if context.function is None:
            raise ParseError.from_node(match.node, context, "function lacks parser context")
        if context.function.dialect != "hir":
            return value
        declared = value.return_annotation
        body_type = context.function.state.body_type
        try:
            if body_type is None:
                if context.function.role is FunctionRole.ROOT and declared is None:
                    raise ParseError.from_node(
                        match.node,
                        context,
                        "HIR pass prototype requires a return annotation",
                    )
                return value
            if declared is not None and not utils.types_compatible(declared, body_type):
                raise ParseError.from_node(
                    match.node,
                    context,
                    "return annotation is not compatible with the inferred body type: "
                    f"annotation {declared!r}, body {body_type!r}",
                )
            if context.function.role is FunctionRole.VARIANT:
                base = context.function.base
                if not isinstance(base, runtime.Function):
                    raise ParseError.from_node(
                        match.node, context, "variant lacks a HIR dispatch prototype"
                    )
                if not utils.types_compatible(base.return_annotation, body_type):
                    raise ParseError.from_node(
                        match.node,
                        context,
                        f"variant body type {body_type!r} is not compatible with "
                        f"dispatch return contract {base.return_annotation!r}",
                    )
            return value
        finally:
            context.function.state.forget_body_type()


def validate_standalone(
    function: object,
    function_context: FuncContext,
    node: ast.AST,
    match_context: MatchContext,
) -> None:
    if function_context.role is FunctionRole.ROOT:
        return
    base = function_context.base
    expected_base = runtime.Function if function_context.dialect == "hir" else runtime.PrimFunction
    if not isinstance(base, expected_base):
        raise ParseError.from_node(node, match_context, "standalone role lacks a matching base")
    if getattr(base, "sealed", False):
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


@dataclass(frozen=True)
class FunctionRoleValidationRule:
    STATEMENT: ClassVar[str] = (
        "A root, variant, or converter must satisfy its role before registration."
    )

    def apply(self, value, *, match, context):
        if context.function is None:
            raise ParseError.from_node(match.node, context, "function lacks parser context")
        if context.function.module is not None:
            context.function.module.validate_function(value, context.function)
        else:
            validate_standalone(value, context.function, match.node, context)
        return value


def commit_standalone(
    function: object,
    function_context: FuncContext,
) -> None:
    if function_context.role is FunctionRole.VARIANT:
        assert function_context.base is not None
        function_context.base.add_variant(function)
    elif function_context.role is FunctionRole.CONVERTER:
        assert function_context.base is not None
        function_context.base.add_converter(function_context.key, function)


@dataclass(frozen=True)
class FunctionRegistrationRule:
    STATEMENT: ClassVar[str] = (
        "A validated function must be registered exactly once in its owning scope."
    )

    def apply(self, value, *, match, context):
        if context.function is None:
            raise ParseError.from_node(match.node, context, "function lacks parser context")
        if context.function.module is not None:
            context.function.module.commit_function(value, context.function)
        else:
            commit_standalone(value, context.function)
        return value


def enter_function(
    node: object, context: MatchContext, matched: AstMatch[Any]
) -> AstMatch[Any] | None:
    assert isinstance(node, ast.FunctionDef)
    function_context = context.function
    if function_context is None:
        return None
    active_context = context.child(
        situation="function",
        role=function_context.function_kind,
        function=function_context,
        live_out=live_out(node.body),
    )
    return dataclasses.replace(
        matched,
        pattern_id="function",
        branch_id="function",
        construct_context=active_context,
    )


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
                        transform=lambda body: body_as_ast_module(body, strip_docstring=True),
                    ),
                ),
            ),
            enter_function,
        )
    )

    def construct(self, match, children, context):
        if context.function is None:
            raise ParseError.from_node(match.node, context, "function lacks context")
        params = children["signature"]
        body = children["body"]
        declared_return = children.get("return")
        specializations = context.function.specializations
        converter = context.function.converter
        if context.function.dialect == "hir":
            if body is not None and context.function.mesh is not None:
                outer_mesh = authored_infer_context(context).current_mesh
                if not isinstance(outer_mesh, runtime.Mesh):
                    raise ParseError.from_node(
                        match.node, context, "function mesh could not be resolved"
                    )
                region_params = tuple(
                    runtime.Var(type=param.type, name=param.name) for param in params
                )
                body = runtime.BindingSubstitutionCloner().visit(
                    body,
                    {id(old): new for old, new in zip(params, region_params, strict=True)},
                )
                body = scoped_region(outer_mesh, body, params=region_params, args=params)
            declared_return = (
                None if declared_return is None else dim.canonicalize_dims(declared_return)
            )
            if body is None:
                return_type = declared_return or runtime.UnitType()
            elif context.function.role is FunctionRole.VARIANT:
                base = context.function.base
                assert isinstance(base, runtime.Function)
                return_type = base.return_type
            else:
                return_type = body.type
            function_name = (
                getattr(context.function.base, "name", None)
                or context.function.base_name
                or match.captures["name"]
            )
            if context.function.role is FunctionRole.CONVERTER:
                function_name = f"{function_name}.converter[{converter}]"
            function = runtime.Function.from_parts(
                name=function_name,
                params=params,
                body=body,
                return_type=return_type,
                return_annotation=declared_return,
                specializations=specializations,
            )
            context.function.state.record_body_type(None if body is None else body.type)
            if context.function.role is FunctionRole.VARIANT:
                function = dataclasses.replace(function, display_name=match.captures["name"])
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
        function_name = match.captures["name"]
        if specializations and isinstance(specializations[0], runtime.Range):
            function_name = pattern.mangle_variant_name(function_name, (specializations[0],))
        function = runtime.PrimFunction(
            name=function_name,
            params=params,
            body=body,
            output_count=context.function.output_count,
            **kwargs,
            specializations=specializations,
            display_name=match.captures["name"],
        )
        define = getattr(context.function.module_scope, "define", None)
        if callable(define):
            define(context.function.binding_name or match.captures["name"], function)
        return function

    RULES: ClassVar[tuple[AstRule[Any], ...]] = (
        FunctionSignatureRule(),
        FunctionDialectRule(),
        FunctionReturnCompatibilityRule(),
        FunctionRoleValidationRule(),
        FunctionRegistrationRule(),
    )


def body_as_ast_module(body: object, *, strip_docstring: bool = False) -> ast.Module:
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
    "CallPattern",
    "CallTypeInferenceRule",
    "ConstantPattern",
    "DTypePattern",
    "DimExprPattern",
    "ExpressionPattern",
    "ForPattern",
    "FunctionDialectRule",
    "FunctionPattern",
    "FunctionRegistrationRule",
    "FunctionReturnCompatibilityRule",
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
    "TupleTypePattern",
    "TypeAnnotationPattern",
    "UnaryExpressionPattern",
    "CallVariadicInputFormRule",
    "VariadicInputs",
    "VariadicInputsPattern",
    "WithPattern",
]
