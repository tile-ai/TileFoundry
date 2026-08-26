"""Executable AST patterns for the parser rewrite prototype.

Patterns choose one local grammar branch. The resulting :class:`AstMatch`
constructs real TileFoundry values after its declared children have been
constructed, then applies the owning pattern's immutable rules in order.
"""

# ruff: noqa: PLC0415, D202, E402, F403, F405

from __future__ import annotations

import ast
import dataclasses
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import FrameType, SimpleNamespace
from typing import Any, ClassVar, Generic, Protocol, TypeVar
from typing import Literal as TypingLiteral

from tilefoundry.ir.core import (
    BindingMetadata,
    Call,
    Constant,
    ExecutionDomainMetadata,
    Expr,
    SourceSpanMetadata,
    Var,
    VerifyError,
    replace_metadata,
)
from tilefoundry.ir.core.expr import Tuple as IrTuple
from tilefoundry.ir.core.kinds import BinaryKind, UnaryKind
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.op_schema import OpSchema
from tilefoundry.ir.hir.function import Function, elaborate
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.sharding.local import Local
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.specialize import DISPLAY_NAME
from tilefoundry.ir.hir.tensor.arange import Arange
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice, slice_size
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, LetStmt, MeshScope, Return, Sequential
from tilefoundry.ir.types import DType, TensorType, TupleType, UnitType
from tilefoundry.ir.types.dim import (
    DimAdd,
    DimFloorDiv,
    DimMod,
    DimMul,
    DimSub,
    DimVar,
    dim_expr,
    simplify_dim,
)
from tilefoundry.ir.types.dim_isl import normalize_dim
from tilefoundry.ir.types.shard import (
    Broadcast,
    Layout,
    Mesh,
    ShardLayout,
    Split,
    c_order_strides,
    canonical_shard_layout,
    composed,
)
from tilefoundry.ir.types.shard.layout import LayoutBase
from tilefoundry.ir.types.storage import StorageKind, resolve_storage
from tilefoundry.visitor_registry.contexts import CallFeed, FunctionScope, TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor

T = TypeVar("T")
_RETURN_TYPE = "<return_type>"
_TYPE_INFER_CONTEXT = "<type_infer_context>"


@dataclass(frozen=True)
class PatternFailure:
    """A recognized pattern whose nested validation failed."""

    pattern_id: str
    node: object
    detail: str
    causes: tuple["PatternFailure", ...] = ()

    def render(self, indent: int = 0) -> str:
        lines = [f"{'  ' * indent}{self.pattern_id}: {self.detail}"]
        for cause in self.causes:
            lines.append(cause.render(indent + 1))
        return "\n".join(lines)


def _pattern_label(pattern: object) -> str:
    label = getattr(pattern, "element_name", None)
    if isinstance(label, str) and label:
        return label
    pattern_id = getattr(pattern, "pattern_id", None)
    if isinstance(pattern_id, str) and pattern_id:
        return pattern_id
    node_type = getattr(pattern, "node_type", None)
    if isinstance(node_type, type):
        return node_type.__name__
    return type(pattern).__name__


def _wrap_failure(pattern: object, node: object, failure: PatternFailure) -> PatternFailure:
    return PatternFailure(
        pattern_id=_pattern_label(pattern),
        node=node,
        detail="nested pattern failed",
        causes=(failure,),
    )


def attach_authored_metadata(value: object, node: ast.AST, context: "MatchContext") -> object:
    """Attach source identity without copying lexical values such as parameter Vars."""
    if not isinstance(value, Expr):
        return value
    owner = context.function or context.module
    source_filename = owner.source_filename if owner is not None else "<string>"
    line = getattr(node, "lineno", None)
    column = getattr(node, "col_offset", None)
    if isinstance(line, int) and isinstance(column, int):
        value = replace_metadata(
            value,
            SourceSpanMetadata(
                source_filename,
                line,
                column + 1,
                getattr(node, "end_lineno", None),
                getattr(node, "end_col_offset", None),
            ),
        )
    if context.binding_name:
        value = replace_metadata(value, BindingMetadata(context.binding_name))
    return value


runtime = SimpleNamespace(
        Call=Call,
        Broadcast=Broadcast,
        Binary=Binary,
        BinaryKind=BinaryKind,
        Constant=Constant,
        DType=DType,
        DimAdd=DimAdd,
        DimFloorDiv=DimFloorDiv,
        DimMod=DimMod,
        DimMul=DimMul,
        DimSub=DimSub,
        DimVar=DimVar,
        Evaluate=Evaluate,
        Expr=Expr,
        ExecutionDomainMetadata=ExecutionDomainMetadata,
        Function=Function,
        GridRegionExpr=GridRegionExpr,
        Arange=Arange,
        IrTuple=IrTuple,
        Layout=Layout,
        LayoutBase=LayoutBase,
        Local=Local,
        LetStmt=LetStmt,
        MeshScope=MeshScope,
        Mesh=Mesh,
        Module=Module,
        OpSchema=OpSchema,
        PrimFunction=PrimFunction,
        Reshard=Reshard,
        Reshape=Reshape,
        Return=Return,
        Sequential=Sequential,
        ShardLayout=ShardLayout,
        Slice=Slice,
        Split=Split,
        StorageKind=StorageKind,
        TensorType=TensorType,
        TupleType=TupleType,
        TypeInferContext=TypeInferContext,
        FunctionScope=FunctionScope,
        TypeInferVisitor=TypeInferVisitor,
        TupleGetItem=TupleGetItem,
        Unary=Unary,
        UnaryKind=UnaryKind,
        UnitType=UnitType,
        Var=Var,
        DISPLAY_NAME=DISPLAY_NAME,
        c_order_strides=c_order_strides,
        canonical_shard_layout=canonical_shard_layout,
        composed=composed,
        dim_expr=dim_expr,
        elaborate=elaborate,
        normalize_dim=normalize_dim,
        slice_size=slice_size,
        simplify_dim=simplify_dim,
        resolve_storage=resolve_storage,
    )


class AstRule(Protocol[T]):
    STATEMENT: ClassVar[str]

    def apply(
        self,
        value: T,
        *,
        match: AstMatch[T],
        context: MatchContext,
    ) -> T: ...


class AstPattern(Protocol[T]):
    element_name: str | None

    def accept(self, visitor: PatternVisitor[Any]) -> Any: ...

    def match(
        self, node: object, context: MatchContext
    ) -> AstMatch[T] | PatternFailure | None: ...


class PatternVisitor(Protocol[T]):
    def visit(self, pattern: AstPattern[Any]) -> T: ...


class CombinatorPattern(AstPattern[Any]):
    """Shared base for executable AstPattern composition nodes."""

    RULES: ClassVar[tuple[AstRule[Any], ...]] = ()
    element_name: str | None = None

    def accept(self, visitor: PatternVisitor[T]) -> T:
        return visitor.visit(self)

    @staticmethod
    def construct(
        match: AstMatch[Any], children: Mapping[str, object], context: MatchContext
    ) -> object:
        if children:
            return tuple(children.values())
        return match.captures.get("value")

    @staticmethod
    def _merge(
        owner: AstPattern[Any],
        node: object,
        matches: tuple[AstMatch[Any], ...],
        *,
        pattern_id: str,
        branch_id: str,
    ) -> AstMatch[Any]:
        captures: dict[str, object] = {}
        children: list[AstChild] = []
        construct_context = None
        selected_branch = branch_id
        selected_pattern_id = pattern_id
        structural_ids = {
            "sequence",
            "field",
            "repeat",
            "optional",
            "capture",
            "child",
            "predicate",
        }
        for matched in matches:
            captures.update(matched.captures)
            children.extend(matched.children)
            construct_context = matched.construct_context or construct_context
            if matched.branch_id not in structural_ids:
                selected_branch = matched.branch_id
            if matched.pattern_id not in structural_ids:
                selected_pattern_id = matched.pattern_id
        return AstMatch(
            owner,
            selected_pattern_id,
            node,
            captures,
            selected_branch,
            tuple(children),
            construct_context,
        )


class ElementPattern(CombinatorPattern, Generic[T]):
    """A named grammar production backed by one executable syntax graph."""

    syntax: ClassVar[AstPattern[Any] | None] = None

    def match(self, node: object, context: MatchContext) -> AstMatch[T] | None:
        syntax = type(self).syntax
        if syntax is None:
            raise TypeError(f"{type(self).__name__} has no executable syntax")
        matched = syntax.match(node, context)
        if isinstance(matched, PatternFailure):
            return _wrap_failure(self, node, matched)
        if matched is None:
            return None
        if isinstance(matched.pattern, ElementPattern):
            return matched
        return AstMatch(
            self,
            matched.pattern_id,
            node,
            matched.captures,
            matched.branch_id,
            matched.children,
            matched.construct_context,
        )


class LazyPattern(CombinatorPattern):
    """Resolve one statically bound forward or recursive Pattern reference."""

    def __init__(self, factory: Callable[[], AstPattern[Any]]):
        self.factory = factory
        self._resolved: AstPattern[Any] | None = None

    @property
    def pattern(self) -> AstPattern[Any]:
        if self._resolved is None:
            self._resolved = self.factory()
        return self._resolved

    def match(
        self, node: object, context: MatchContext
    ) -> AstMatch[Any] | PatternFailure | None:
        matched = self.pattern.match(node, context)
        return matched


class AstNodePattern(CombinatorPattern):
    def __init__(self, node_type: type, *parts: AstPattern[Any]):
        self.node_type = node_type
        self.parts = tuple(parts)

    def match(self, node: object, context: MatchContext) -> AstMatch[Any] | PatternFailure | None:
        if not isinstance(node, self.node_type):
            return None
        matches: list[AstMatch[Any]] = []
        for part in self.parts:
            matched = part.match(node, context)
            if isinstance(matched, PatternFailure):
                return _wrap_failure(self, node, matched)
            if matched is None:
                return None
            matches.append(matched)
        return self._merge(
            self,
            node,
            tuple(matches),
            pattern_id=self.node_type.__name__,
            branch_id=self.node_type.__name__.lower(),
        )


class FieldPattern(CombinatorPattern):
    def __init__(self, name: str, pattern: AstPattern[Any]):
        self.name = name
        self.pattern = pattern

    def match(self, node: object, context: MatchContext) -> AstMatch[Any] | PatternFailure | None:
        if not hasattr(node, self.name):
            return None
        value = getattr(node, self.name)
        matched = self.pattern.match(value, context)
        if isinstance(matched, PatternFailure):
            return _wrap_failure(self, node, matched)
        if matched is None:
            return None
        return AstMatch(
            self,
            matched.pattern_id,
            node,
            matched.captures,
            matched.branch_id,
            matched.children,
            matched.construct_context,
        )


class LiteralPattern(CombinatorPattern):
    def __init__(
        self,
        value: object = dataclasses.MISSING,
        *,
        value_type: type | tuple[type, ...] | None = None,
    ):
        self.value = value
        self.value_type = value_type

    def match(self, node: object, context: MatchContext) -> AstMatch[Any] | PatternFailure | None:
        raw = node.value if isinstance(node, ast.Constant) else node
        if self.value is not dataclasses.MISSING and raw != self.value:
            return None
        if self.value_type is not None and not isinstance(raw, self.value_type):
            return None
        return AstMatch(self, "literal", node, {"value": raw}, "literal")


class ReferencePattern(CombinatorPattern):
    def __init__(
        self, *, resolve: bool = False, expected: type | tuple[type, ...] | None = None
    ):
        self.resolve = resolve
        self.expected = expected

    def match(self, node: object, context: MatchContext) -> AstMatch[Any] | PatternFailure | None:
        if not isinstance(node, (ast.Name, ast.Attribute)):
            return None
        captures: dict[str, object] = {}
        if self.resolve:
            try:
                value = _resolve_reference(node, context)
            except ParseError:
                return None
            if self.expected is not None and not isinstance(value, self.expected):
                return None
            captures["reference"] = value
        return AstMatch(self, "reference", node, captures, "reference")


class SequencePattern(CombinatorPattern):
    def __init__(self, *patterns: AstPattern[Any]):
        self.patterns = tuple(patterns)

    def match(self, node: object, context: MatchContext) -> AstMatch[Any] | PatternFailure | None:
        if not isinstance(node, (tuple, list)) or len(node) != len(self.patterns):
            return None
        matches: list[AstMatch[Any]] = []
        for value, pattern in zip(node, self.patterns):
            matched = pattern.match(value, context)
            if isinstance(matched, PatternFailure):
                return _wrap_failure(self, node, matched)
            if matched is None:
                return None
            matches.append(matched)
        return self._merge(
            self,
            node,
            tuple(matches),
            pattern_id="sequence",
            branch_id="sequence",
        )


class ChoicePattern(CombinatorPattern):
    def __init__(self, *patterns: AstPattern[Any]):
        self.patterns = tuple(patterns)

    def match(self, node: object, context: MatchContext) -> AstMatch[Any] | PatternFailure | None:
        failures: list[PatternFailure] = []
        for pattern in self.patterns:
            matched = pattern.match(node, context)
            if isinstance(matched, PatternFailure):
                failures.append(matched)
                continue
            if matched is not None:
                return matched
            failures.append(
                PatternFailure(
                    pattern_id=_pattern_label(pattern),
                    node=node,
                    detail="pattern did not match",
                )
            )
        if failures:
            return PatternFailure(
                pattern_id="choice",
                node=node,
                detail="no choice matched",
                causes=tuple(failures),
            )
        return None


class ConditionPattern(CombinatorPattern):
    """Run one sub-pattern only when an explicit context condition holds."""

    def __init__(
        self,
        label: str,
        test: Callable[[object, MatchContext], bool],
        pattern: AstPattern[Any],
    ):
        self.label = label
        self.test = test
        self.pattern = pattern

    def match(self, node: object, context: MatchContext) -> AstMatch[Any] | PatternFailure | None:
        if not self.test(node, context):
            return None
        return self.pattern.match(node, context)


class OptionalPattern(CombinatorPattern):
    def __init__(self, pattern: AstPattern[Any]):
        self.pattern = pattern

    def match(self, node: object, context: MatchContext) -> AstMatch[Any] | PatternFailure | None:
        if node is None:
            return AstMatch(self, "optional", node, {}, "optional")
        matched = self.pattern.match(node, context)
        if isinstance(matched, PatternFailure):
            return _wrap_failure(self, node, matched)
        if matched is None:
            return None
        return AstMatch(
            self,
            matched.pattern_id,
            node,
            matched.captures,
            matched.branch_id,
            matched.children,
            matched.construct_context,
        )


class RepeatPattern(CombinatorPattern):
    def __init__(self, pattern: AstPattern[Any], *, minimum: int = 0):
        self.pattern = pattern
        self.minimum = minimum

    @staticmethod
    def _index_child(child: AstChild, index: int) -> AstChild:
        return dataclasses.replace(child, name=child.name.format(index=index))

    def match(self, node: object, context: MatchContext) -> AstMatch[Any] | PatternFailure | None:
        if not isinstance(node, (tuple, list)) or len(node) < self.minimum:
            return None
        matches: list[AstMatch[Any]] = []
        for index, value in enumerate(node):
            matched = self.pattern.match(value, context)
            if isinstance(matched, PatternFailure):
                return _wrap_failure(self, node, matched)
            if matched is None:
                return None
            matches.append(
                dataclasses.replace(
                    matched,
                    children=tuple(
                        self._index_child(child, index) for child in matched.children
                    ),
                )
            )
        return self._merge(
            self,
            node,
            tuple(matches),
            pattern_id="repeat",
            branch_id="repeat",
        )


class PredicatePattern(CombinatorPattern):
    def __init__(self, label: str, predicate: Callable[[object, MatchContext], bool]):
        self.label = label
        self.predicate = predicate

    def match(self, node: object, context: MatchContext) -> AstMatch[Any] | PatternFailure | None:
        if not self.predicate(node, context):
            return None
        return AstMatch(self, "predicate", node, {}, "predicate")


class CapturePattern(CombinatorPattern):
    def __init__(self, name: str, extractor: Callable[[object, MatchContext], object]):
        self.name = name
        self.extractor = extractor

    def match(self, node: object, context: MatchContext) -> AstMatch[Any] | PatternFailure | None:
        try:
            value = self.extractor(node, context)
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        return AstMatch(self, "capture", node, {self.name: value}, "capture")


class ChildPattern(CombinatorPattern):
    def __init__(
        self,
        name: str,
        pattern: AstPattern[Any] | Callable[[], AstPattern[Any]],
        situation: str,
        role: str | None = None,
        *,
        expected_type: object | Callable[[object, MatchContext], object] | None = None,
        values: Mapping[str, object]
        | Callable[[object, MatchContext], Mapping[str, object]]
        | None = None,
        isolated_scope: bool = False,
        transform: Callable[[object], object] | None = None,
    ):
        self.name = name
        self._pattern = pattern
        self.situation = situation
        self.role = role
        self.expected_type = expected_type
        self.values = values
        self.isolated_scope = isolated_scope
        self.transform = transform

    @property
    def pattern(self) -> AstPattern[Any]:
        pattern = self._pattern
        if callable(pattern):
            pattern = pattern()
            self._pattern = pattern
        return pattern

    def match(self, node: object, context: MatchContext) -> AstMatch[Any] | PatternFailure | None:
        value = self.transform(node) if self.transform is not None else node
        if not isinstance(value, ast.AST):
            return None
        values = (
            self.values(value, context) if callable(self.values) else self.values or {}
        )
        expected_type = (
            self.expected_type(value, context)
            if callable(self.expected_type)
            else self.expected_type
        )
        child = AstChild(
            self.name,
            self.pattern,
            value,
            self.situation,
            self.role,
            expected_type=expected_type,
            values=values,
            isolated_scope=self.isolated_scope,
        )
        return AstMatch(self, "child", node, {}, "child", (child,))


class BranchPattern(CombinatorPattern):
    def __init__(
        self, branch_id: str, pattern: AstPattern[Any], *, pattern_id: str | None = None
    ):
        self.branch_id = branch_id
        self.pattern = pattern
        self.pattern_id = pattern_id or branch_id

    def match(
        self, node: object, context: MatchContext
    ) -> AstMatch[Any] | PatternFailure | None:
        matched = self.pattern.match(node, context)
        if isinstance(matched, PatternFailure):
            return _wrap_failure(self, node, matched)
        if matched is None:
            return None
        return AstMatch(
            self,
            self.pattern_id,
            node,
            matched.captures,
            self.branch_id,
            matched.children,
            matched.construct_context,
        )


class BindPattern(CombinatorPattern):
    """Add semantic captures/children after an executable structural match."""

    def __init__(
        self,
        pattern: AstPattern[Any],
        binder: Callable[[object, MatchContext, AstMatch[Any]], AstMatch[Any] | None],
    ):
        self.pattern = pattern
        self.binder = binder

    def match(self, node: object, context: MatchContext) -> AstMatch[Any] | PatternFailure | None:
        matched = self.pattern.match(node, context)
        if isinstance(matched, PatternFailure):
            return _wrap_failure(self, node, matched)
        if matched is None:
            return None
        bound = self.binder(node, context, matched)
        return bound


class LexicalScope:
    """Parser-local lexical frames shared by sequential child construction."""

    def __init__(self, frames: tuple[Mapping[str, object], ...] | None = None):
        source = frames or ({},)
        self._frames = [dict(frame) for frame in source]

    def define(self, name: str, value: object) -> None:
        self._frames[-1][name] = value

    def lookup(self, name: str) -> object | None:
        for frame in reversed(self._frames):
            if name in frame:
                return frame[name]
        return None

    def fork(self) -> LexicalScope:
        return LexicalScope(tuple(self._frames) + ({},))

    def push_frame(self) -> None:
        self._frames.append({})

    def pop_frame(self) -> dict[str, object]:
        if len(self._frames) == 1:
            raise RuntimeError("cannot pop the root lexical frame")
        return self._frames.pop()

    def items(self):
        merged: dict[str, object] = {}
        for frame in self._frames:
            merged.update(frame)
        return merged.items()


class FunctionRole(StrEnum):
    ROOT = "root"
    VARIANT = "variant"
    CONVERTER = "converter"


@dataclass
class ParserState:
    """Mutable state owned by one Function parse."""

    mesh_stack: list[object] = field(default_factory=list)
    mesh_coordinates: dict[tuple[int, int], object] = field(default_factory=dict)


@dataclass(frozen=True)
class LoopFrame:
    kind: str
    target: str
    induction_var: object
    start: object
    extent: object
    step: object
    carry_names: tuple[str, ...]
    phi_vars: tuple[object, ...]
    init_args: tuple[object, ...]


@dataclass(frozen=True)
class FuncParserContext:
    """Function-level authored inputs known before walking a FunctionDef."""

    dialect: TypingLiteral["hir", "tir"]
    role: FunctionRole = FunctionRole.ROOT
    closure: Mapping[str, object] = field(default_factory=dict)
    topologies: Mapping[str, object] = field(default_factory=dict)
    source_filename: str = "<string>"
    module_scope: object | None = None
    module: ModuleBuildContext | None = None
    base: object | None = None
    key: object | None = None
    target: object | None = None
    output_count: int = 1
    hardware_context: Mapping[str, object] = field(default_factory=dict)
    binding_name: str | None = None
    base_name: str | None = None
    function_kind: str | None = field(default=None, repr=False)
    state: ParserState = field(default_factory=ParserState)

    def __post_init__(self) -> None:
        role = self.role
        if self.function_kind is not None and role is FunctionRole.ROOT:
            role = self.function_kind
        if isinstance(role, str) and role not in {item.value for item in FunctionRole}:
            role = (
                FunctionRole.ROOT
                if role in {"func", "prim_func", "kernel"}
                else FunctionRole(role)
            )
        if role is not self.role:
            object.__setattr__(self, "role", role)
        if self.function_kind is None:
            object.__setattr__(
                self,
                "function_kind",
                "prim_func"
                if role is FunctionRole.ROOT and self.dialect == "tir"
                else "func"
                if role is FunctionRole.ROOT
                else role.value,
            )

    @property
    def specializations(self) -> tuple[object, ...]:
        return (
            ()
            if self.role is not FunctionRole.VARIANT or self.key is None
            else (self.key,)
        )

    @property
    def converter(self) -> object | None:
        return self.key if self.role is FunctionRole.CONVERTER else None


@dataclass(frozen=True)
class ParserCallFeedProvider:
    """Build call feeds from the authored module scope during parsing."""

    module_scope: object | None

    def _child_for(self, callee: object):
        if self.module_scope is None:
            return None
        for _name, child in self.module_scope.items():
            if getattr(child, "owns", lambda *_args, **_kwargs: False)(
                callee, derived=True
            ):
                return child
        return None

    def build_call_feed(self, callee: object, supplied: tuple[object, ...]) -> CallFeed:
        child = self._child_for(callee)
        params = tuple(p for p in callee.params if not (child is not None and p.is_const))
        if len(supplied) != len(params):
            kind = "activation(s)" if child is not None else "parameter(s)"
            raise VerifyError(
                f"hir Function call {callee.name!r}: arity mismatch — "
                f"callee declares {len(params)} {kind}, call passed {len(supplied)}"
            )
        given = iter(supplied)
        return CallFeed(
            {
                id(param): param.type if child is not None and param.is_const else next(given)
                for param in callee.params
            }
        )

    def scope_for(self, callee: object) -> FunctionScope | None:
        child = self._child_for(callee)
        return None if child is None else FunctionScope(child, callee)


@dataclass(frozen=True)
class ModuleFunctionValidationRule:
    STATEMENT: ClassVar[str] = (
        "A module function must satisfy its root, variant, or converter role before mutation."
    )

    def apply(
        self,
        function: object,
        *,
        context: FuncParserContext,
        module: ModuleBuildContext,
    ) -> object:
        module._validate_function(function, context)
        return function


@dataclass(frozen=True)
class ModuleFunctionRegistrationRule:
    STATEMENT: ClassVar[str] = (
        "A validated module function must be recorded in declaration order."
    )

    def apply(
        self,
        function: object,
        *,
        context: FuncParserContext,
        module: ModuleBuildContext,
    ) -> object:
        module._commit_function(function, context)
        return function


@dataclass(frozen=True)
class ModuleFinalizationRule:
    STATEMENT: ClassVar[str] = (
        "A module declaration must contain valid unique members and a resolvable entry."
    )

    def apply(self, cls: type, *, module: ModuleBuildContext) -> object:
        return module._finalize(cls)


@dataclass
class ModuleBuildContext:
    """Authoring ledger for one Python ``@module`` class body.

    The context is deliberately independent from recursive ``MatchContext``.
    It is created while the decorator expression is evaluated, looked up by
    declaring frame during class-body execution, and consumed exactly once by
    :meth:`finalize`.
    """

    owner_frame: FrameType
    owner_name: str | None = None
    closure: Mapping[str, object] = field(default_factory=dict)
    source_filename: str = "<string>"
    module_scope: LexicalScope = field(default_factory=LexicalScope)
    entry: str | None = None
    target: object | None = None
    topologies: tuple[object, ...] | None = None
    roots: list[object] = field(default_factory=list)
    bindings: dict[str, FunctionRole] = field(default_factory=dict)
    binding_values: dict[str, object] = field(default_factory=dict)
    owned: dict[int, FunctionRole] = field(default_factory=dict)
    variant_keys: dict[int, set[object]] = field(default_factory=dict)
    converter_keys: dict[int, set[str]] = field(default_factory=dict)
    declarations: list[object] = field(default_factory=list)
    _consumed: bool = False
    FUNCTION_RULES: ClassVar[tuple[object, ...]] = (
        ModuleFunctionValidationRule(),
        ModuleFunctionRegistrationRule(),
    )
    FINALIZATION_RULES: ClassVar[tuple[object, ...]] = (ModuleFinalizationRule(),)

    def function_context(
        self,
        *,
        dialect: TypingLiteral["hir", "tir"],
        role: FunctionRole,
        binding_name: str,
        closure: Mapping[str, object] | None = None,
        base: object | None = None,
        key: object | None = None,
    ) -> FuncParserContext:
        topology_scope = {
            getattr(topology, "name", str(index)): topology
            for index, topology in enumerate(self.topologies or ())
        }
        return FuncParserContext(
            dialect=dialect,
            role=role,
            closure=closure if closure is not None else self.closure,
            topologies=topology_scope,
            source_filename=self.source_filename,
            module_scope=self.module_scope,
            module=self,
            base=base,
            key=key,
            target=self.target if dialect == "tir" else None,
            binding_name=binding_name,
        )

    @staticmethod
    def _binding_error(role: FunctionRole, binding: str, owner: str | None) -> str:
        return f"@module {owner or '<module>'!r}: duplicate {role.value} binding {binding!r}"

    def validate_function(self, function: object, context: FuncParserContext) -> None:
        self.FUNCTION_RULES[0].apply(function, context=context, module=self)

    def _validate_function(self, function: object, context: FuncParserContext) -> None:
        role = context.role
        binding = context.binding_name or getattr(function, "name", "<anonymous>")
        if role is FunctionRole.ROOT:
            if binding == "_":
                raise ValueError(
                    f"@module {self.owner_name or '<module>'!r}: a root binding may not be named '_'"
                )
            if binding in self.bindings:
                raise ValueError(self._binding_error(role, binding, self.owner_name))
            if any(
                getattr(root, "name", None) == getattr(function, "name", None)
                for root in self.roots
            ):
                raise ValueError(
                    self._binding_error(
                        role, getattr(function, "name", binding), self.owner_name
                    )
                )
            expected = (
                runtime.PrimFunction if context.dialect == "tir" else runtime.Function
            )
            if not isinstance(function, expected):
                raise TypeError(
                    f"root {binding!r} constructed {type(function).__name__}, expected {expected.__name__}"
                )
            return
        base = context.base
        if base is None or not isinstance(base, runtime.Function):
            raise ValueError(f"{role.value} {binding!r}: base is not a HIR Function")
        if getattr(base, "_sealed", False):
            raise RuntimeError(
                f"base {base.name!r}: cannot register {role.value} after seal"
            )
        if binding == "_" and role is FunctionRole.VARIANT:
            raise ValueError(
                f"base {base.name!r}: a variant binding may not be named '_'"
            )
        if binding in self.bindings and role is not FunctionRole.CONVERTER:
            raise ValueError(self._binding_error(role, binding, self.owner_name))
        if role is FunctionRole.VARIANT:
            if getattr(function, "body", None) is None:
                raise ValueError(f"base {base.name!r}: a variant must have a real body")
            keys = self.variant_keys.setdefault(id(base), set())
            key = context.key
            if key in keys:
                raise ValueError(
                    f"base {base.name!r}: duplicate specialization key {key!r}"
                )
            return
        if getattr(function, "body", None) is None:
            raise ValueError(f"base {base.name!r}: a converter must have a real body")
        key = context.key
        if not isinstance(key, str):
            raise TypeError(f"base {base.name!r}: converter weight key must be str")
        keys = self.converter_keys.setdefault(id(base), set())
        if key in keys:
            raise ValueError(f"base {base.name!r}: duplicate converter weight {key!r}")

    def commit_function(self, function: object, context: FuncParserContext) -> None:
        self.FUNCTION_RULES[1].apply(function, context=context, module=self)

    def _commit_function(self, function: object, context: FuncParserContext) -> None:
        role = context.role
        binding = context.binding_name or getattr(function, "name", "<anonymous>")
        if role is FunctionRole.ROOT:
            self.roots.append(function)
            self.bindings[binding] = role
            self.binding_values[binding] = function
            return
        base = context.base
        assert base is not None
        if role is FunctionRole.VARIANT:
            base.add_variant(function)
            self.variant_keys.setdefault(id(base), set()).add(context.key)
        else:
            base.add_converter(context.key, function)
            self.converter_keys.setdefault(id(base), set()).add(context.key)
        self.bindings[binding] = role
        self.binding_values[binding] = function
        self.owned[id(function)] = role

    def finalize(self, cls: type) -> object:
        value: object = cls
        for rule in self.FINALIZATION_RULES:
            value = rule.apply(value, module=self)
        return value

    def _finalize(self, cls: type) -> object:
        if self._consumed:
            raise RuntimeError(
                f"@module {self.owner_name or cls.__name__!r}: declaration context already consumed"
            )
        self._consumed = True
        consume_module_context(self)
        functions: list[object] = []
        modules: list[object] = []
        module_bindings: dict[str, object] = {}
        methods: dict[str, object] = {}
        for name, value in vars(cls).items():
            if name == "__call__":
                raise TypeError(
                    f"@module {cls.__name__!r}: a class-body __call__ has no effect "
                    "-- name the method `forward` instead"
                )
            if name.startswith("__") and name.endswith("__"):
                continue
            if isinstance(value, runtime.Module):
                child = value if value.name == name else value.renamed(name)
                modules.append(child)
                module_bindings[name] = child
            elif (
                isinstance(value, (tuple, list))
                and value
                and all(isinstance(item, runtime.Module) for item in value)
            ):
                modules.extend(value)
                for child in value:
                    module_bindings[child.name] = child
            elif isinstance(value, (runtime.Function, runtime.PrimFunction)):
                if id(value) in self.owned:
                    continue
                functions.append(value)
            elif getattr(value, "_tilefoundry_deferred", False):
                continue
            elif callable(value):
                methods[name] = value
            else:
                raise TypeError(
                    f"@module {cls.__name__!r}: member {name!r} is a "
                    f"{type(value).__name__}, not an @func / @prim_func result, a "
                    "Module (or tuple/list of Modules), or a plain function; a "
                    "@module class body may contain only these three member kinds"
                )
        for binding, role in self.bindings.items():
            value = vars(cls).get(binding)
            expected = self.binding_values.get(binding)
            if role is not FunctionRole.CONVERTER and value is not expected:
                raise ValueError(
                    f"@module {cls.__name__!r}: registered {role.value} {binding!r} "
                    "was overwritten or aliased"
                )
        for name, child in module_bindings.items():
            self.module_scope.define(name, child)
        for declaration in self.declarations:
            parsed = declaration.parse()
            if declaration.role is FunctionRole.ROOT:
                functions.append(parsed)

        names = [getattr(fn, "name", None) for fn in functions]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"@module {cls.__name__!r}: duplicate function name(s) {duplicates} "
                "(a class-body alias of a DSL function is not allowed)"
            )
        if not functions and not modules and not methods:
            raise TypeError(
                f"@module {cls.__name__!r}: empty class body; declare a Function, "
                "child Module, or orchestration method"
            )
        if self.entry is not None and self.entry not in names:
            raise ValueError(
                f"@module {cls.__name__!r}: entry {self.entry!r} names no collected function (have {names})"
            )
        result = runtime.Module(
            name=cls.__name__,
            functions=tuple(functions),
            entry=self.entry,
            modules=tuple(modules),
            target=self.target,
            topologies=self.topologies,
            methods=methods,
        )
        from tilefoundry.ir.hir.verify import verify_function  # noqa: PLC0415
        from tilefoundry.ir.tir.verify import verify_prim_function  # noqa: PLC0415

        prim_functions = tuple(
            function for function in functions if isinstance(function, runtime.PrimFunction)
        )
        for function in functions:
            if isinstance(function, runtime.Function):
                verify_function(function, module=result)
            elif isinstance(function, runtime.PrimFunction):
                verify_prim_function(function, module_fns=prim_functions)
        return result


_MODULE_CONTEXTS: list[ModuleBuildContext] = []


def register_module_context(context: ModuleBuildContext) -> None:
    _MODULE_CONTEXTS.append(context)


def create_module_context(
    *,
    entry: str | None = None,
    target: object | None = None,
    topologies: tuple[object, ...] | None = None,
    closure: Mapping[str, object] | None = None,
    owner_frame: FrameType | None = None,
    owner_name: str | None = None,
    source_filename: str = "<string>",
) -> ModuleBuildContext:
    """Create and register a context from a module decorator expression."""

    frame = owner_frame or __import__("sys")._getframe(1)
    context = ModuleBuildContext(
        owner_frame=frame,
        owner_name=owner_name,
        closure=closure or {},
        source_filename=source_filename,
        entry=entry,
        target=target,
        topologies=topologies,
    )
    register_module_context(context)
    return context


def module_context_for_frame(frame: FrameType | None) -> ModuleBuildContext | None:
    while frame is not None:
        if "__qualname__" in frame.f_locals:
            for context in reversed(_MODULE_CONTEXTS):
                if context.owner_frame is frame.f_back:
                    context.owner_name = frame.f_locals["__qualname__"].rsplit(".", 1)[-1]
                    return context
            return None
        if frame.f_code.co_name == "<module>":
            return None
        frame = frame.f_back
    return None


def consume_module_context(context: ModuleBuildContext) -> None:
    try:
        _MODULE_CONTEXTS.remove(context)
    except ValueError:
        pass


@dataclass(frozen=True)
class MatchContext:
    """Inherited context for one recursive grammar position."""

    function: FuncParserContext | None
    module: ModuleBuildContext | None
    situation: str
    role: str | None = None
    binding_name: str | None = None
    expected_type: object | None = None
    lexical_scope: LexicalScope = field(default_factory=LexicalScope)
    parent: MatchContext | None = None
    values: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_function(cls, function: FuncParserContext) -> MatchContext:
        scope = LexicalScope()
        provider = (
            ParserCallFeedProvider(function.module_scope)
            if function.module_scope is not None
            else None
        )
        scope.define(
            _TYPE_INFER_CONTEXT,
            runtime.TypeInferContext(call_feed_provider=provider),
        )
        return cls(
            function=function,
            module=None,
            situation="function",
            role="function",
            lexical_scope=scope,
            values=function.hardware_context,
        )

    def child(
        self,
        *,
        situation: str,
        role: str | None = None,
        binding_name: str | None = None,
        expected_type: object | None = None,
        values: Mapping[str, object] | None = None,
        lexical_bindings: Mapping[str, object] | None = None,
        isolated_scope: bool = False,
        function: FuncParserContext | None = None,
        module: ModuleBuildContext | None = None,
    ) -> MatchContext:
        merged = dict(self.values)
        if values:
            merged.update(values)
        switching_function = function is not None and function is not self.function
        if switching_function:
            scope = LexicalScope()
            provider = (
                ParserCallFeedProvider(function.module_scope)
                if function is not None and function.module_scope is not None
                else None
            )
            scope.define(
                _TYPE_INFER_CONTEXT,
                runtime.TypeInferContext(call_feed_provider=provider),
            )
        else:
            scope = self.lexical_scope.fork() if isolated_scope else self.lexical_scope
        if lexical_bindings:
            for name, value in lexical_bindings.items():
                scope.define(name, value)
        if expected_type is None and role == "return_value":
            expected_type = scope.lookup(_RETURN_TYPE)
        return MatchContext(
            function=function or self.function,
            module=module or self.module,
            situation=situation,
            role=role,
            binding_name=binding_name,
            expected_type=expected_type,
            lexical_scope=scope,
            parent=self,
            values=merged,
        )

    def resolve_lexical(self, name: str) -> object:
        value = self.lexical_scope.lookup(name)
        if value is None:
            raise ParseError.from_node(
                ast.Name(id=name, ctx=ast.Load()),
                self,
                f"undefined lexical name {name!r}",
            )
        return value

    def resolve_static(
        self, node: ast.AST, expected: type[T] | tuple[type[Any], ...]
    ) -> T:
        if not isinstance(node, (ast.Name, ast.Attribute)):
            raise ParseError.from_node(
                node, self, "static references use Name or Attribute"
            )
        value = _resolve_reference(node, self)
        if not isinstance(value, expected):
            raise ParseError.from_node(
                node,
                self,
                f"static reference resolved to {type(value).__name__}, "
                f"expected {_expected_name(expected)}",
            )
        return value


@dataclass(frozen=True)
class AstChild:
    name: str
    pattern: AstPattern[Any]
    node: ast.AST
    situation: str
    role: str | None = None
    expected_type: object | None = None
    values: Mapping[str, object] = field(default_factory=dict)
    isolated_scope: bool = False
    function_context: FuncParserContext | None = None
    module_context: ModuleBuildContext | None = None
    lexical_bindings: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AstMatch(Generic[T]):
    pattern: AstPattern[T]
    pattern_id: str
    node: ast.AST
    captures: Mapping[str, object]
    branch_id: str
    children: tuple[AstChild, ...] = ()
    construct_context: MatchContext | None = None

    def construct(self, children: Mapping[str, object], context: MatchContext) -> T:
        pattern_constructor = getattr(self.pattern, "construct", None)
        if not callable(pattern_constructor):
            raise TypeError(
                f"Pattern {type(self.pattern).__name__} has no construct method"
            )
        value = pattern_constructor(self, children, context)
        for rule in self.pattern.RULES:
            value = rule.apply(value, match=self, context=context)
        return value


class ParseError(VerifyError):
    """A grammar or context rule failure anchored to an authored AST node."""

    def __init__(
        self,
        *,
        node: ast.AST,
        context: MatchContext,
        detail: str | None = None,
    ):
        line = getattr(node, "lineno", None)
        column = getattr(node, "col_offset", None)
        location = ""
        owner = context.function or context.module
        source_filename = owner.source_filename if owner is not None else "<string>"
        if isinstance(line, int):
            location = f" at {source_filename}:{line}"
            if isinstance(column, int):
                location += f":{column + 1}"
        message = detail or f"no AST pattern matched situation {context.situation!r}"
        if context.role:
            message += f" (role {context.role!r})"
        super().__init__(message + location)
        self.node = node
        self.context = context
        self.detail = detail

    @classmethod
    def from_node(
        cls, node: ast.AST, context: MatchContext, detail: str | None = None
    ) -> ParseError:
        return cls(node=node, context=context, detail=detail)


def parse_node(pattern: AstPattern[T], node: ast.AST, context: MatchContext) -> T:
    """Select, recursively construct, and apply rules for one local pattern."""

    matched = pattern.match(node, context)
    if isinstance(matched, PatternFailure):
        raise ParseError.from_node(node, context, matched.render())
    if matched is None:
        raise ParseError.from_node(node, context)
    active_context = matched.construct_context or context
    children: dict[str, object] = {}
    for child in matched.children:
        if child.name in children:
            raise RuntimeError(f"duplicate AstChild name {child.name!r}")
        binding_name = None
        if child.name == "value":
            binding_name = matched.captures.get("name")
            if binding_name is None:
                names = matched.captures.get("names")
                if isinstance(names, tuple):
                    binding_name = ", ".join(names)
        child_context = active_context.child(
            situation=child.situation,
            role=child.role,
            binding_name=binding_name,
            expected_type=child.expected_type,
            values=child.values,
            lexical_bindings=child.lexical_bindings,
            isolated_scope=child.isolated_scope,
            function=child.function_context,
            module=child.module_context,
        )
        children[child.name] = parse_node(child.pattern, child.node, child_context)
    return matched.construct(children, active_context)


def _expected_name(expected: type | tuple[type, ...]) -> str:
    if isinstance(expected, tuple):
        return " | ".join(item.__name__ for item in expected)
    return expected.__name__


def _decorator_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _resolve_reference(node: ast.AST, context: MatchContext) -> object:
    if isinstance(node, ast.Name):
        lexical = context.lexical_scope.lookup(node.id)
        if lexical is not None:
            return lexical
        function = context.function
        module = context.module
        module_scope = (
            function.module_scope
            if function is not None
            else module.module_scope
            if module is not None
            else None
        )
        if isinstance(module_scope, Mapping) and node.id in module_scope:
            return module_scope[node.id]
        lookup = getattr(module_scope, "lookup", None)
        if callable(lookup):
            try:
                value = lookup(node.id)
            except (KeyError, ValueError):
                pass
            else:
                if value is not None:
                    return value
        closure = (
            function.closure
            if function is not None
            else module.closure
            if module is not None
            else {}
        )
        if node.id in closure:
            return closure[node.id]
        raise ParseError.from_node(node, context, f"undefined static name {node.id!r}")
    if isinstance(node, ast.Attribute):
        owner = _resolve_reference(node.value, context)
        try:
            return getattr(owner, node.attr)
        except AttributeError as error:
            raise ParseError.from_node(
                node, context, f"{type(owner).__name__} has no attribute {node.attr!r}"
            ) from error
    raise ParseError.from_node(node, context, "expected static Name or Attribute")


_BINARY_OPERATORS: Mapping[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: Mapping[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}


@dataclass(frozen=True)
class CanonicalDTypeRule:
    STATEMENT: ClassVar[str] = "A dtype must resolve to a canonical DType."

    def apply(self, value, *, match, context):
        if not isinstance(value, runtime.DType):
            raise ParseError.from_node(
                match.node, context, "dtype did not construct DType"
            )
        canonical = runtime.DType._members().get(value.name)
        if canonical is not value:
            raise ParseError.from_node(
                match.node, context, f"non-canonical dtype {value.name!r}"
            )
        return value


@dataclass(frozen=True)
class LayoutShapeRule:
    STATEMENT: ClassVar[str] = "A layout must have a valid non-boolean shape."

    def apply(self, value, *, match, context):
        if value is not None and not isinstance(value, runtime.LayoutBase):
            raise ParseError.from_node(
                match.node, context, "layout did not construct LayoutBase"
            )
        if value is not None and not isinstance(value.shape, tuple):
            raise ParseError.from_node(
                match.node, context, "layout shape is not a tuple"
            )
        return value


@dataclass(frozen=True)
class LayoutPositionRule:
    STATEMENT: ClassVar[str] = "A layout must be legal for its parser position."

    def apply(self, value, *, match, context):
        if context.role in {"storage", "dtype", "shape"}:
            raise ParseError.from_node(
                match.node, context, "layout used in a non-layout role"
            )
        return value


@dataclass(frozen=True)
class StorageValueRule:
    STATEMENT: ClassVar[str] = "Storage must resolve to a StorageKind."

    def apply(self, value, *, match, context):
        if not isinstance(value, runtime.StorageKind):
            raise ParseError.from_node(
                match.node, context, "storage did not construct StorageKind"
            )
        return value


@dataclass(frozen=True)
class TensorLayoutStorageRule:
    STATEMENT: ClassVar[str] = (
        "A tensor type must contain compatible layout and storage values."
    )

    def apply(self, value, *, match, context):
        if not isinstance(value, runtime.TensorType):
            raise ParseError.from_node(
                match.node, context, "tensor did not construct TensorType"
            )
        if not isinstance(value.storage, runtime.StorageKind):
            raise ParseError.from_node(
                match.node, context, "tensor storage is not StorageKind"
            )
        if value.layout is not None and not isinstance(
            value.layout, runtime.LayoutBase
        ):
            raise ParseError.from_node(
                match.node, context, "tensor layout is not LayoutBase"
            )
        return value


@dataclass(frozen=True)
class TensorPositionRule:
    STATEMENT: ClassVar[str] = (
        "A tensor type's storage must be legal for its dialect and position."
    )

    def apply(self, value, *, match, context):
        if context.role == "storage":
            raise ParseError.from_node(
                match.node, context, "TensorType used in a storage role"
            )
        allowed = context.values.get("allowed_storage")
        if allowed is not None and value.storage not in allowed:
            rendered = tuple(str(item) for item in allowed)
            raise ParseError.from_node(
                match.node,
                context,
                f"storage {value.storage} is not allowed by hardware context {rendered}",
            )
        return value


@dataclass(frozen=True)
class ShapeDimRule:
    STATEMENT: ClassVar[str] = (
        "A shape dimension must be an integer, DimVar, or expression."
    )

    def apply(self, value, *, match, context):
        value = runtime.normalize_dim(value)
        if isinstance(value, bool) or not isinstance(
            value, (int, runtime.DimVar, runtime.Expr)
        ):
            raise ParseError.from_node(
                match.node,
                context,
                f"shape dimension must be int, DimVar, or Expr, got {type(value).__name__}",
            )
        return value


@dataclass(frozen=True)
class ShapeTupleRule:
    STATEMENT: ClassVar[str] = "A shape must construct a tuple of dimensions."

    def apply(self, value, *, match, context):
        if not isinstance(value, tuple):
            raise ParseError.from_node(match.node, context, "shape is not a tuple")
        return value


__all__ = [
    "AstChild",
    "AstMatch",
    "AstPattern",
    "AstRule",
    "BlockPattern",
    "CallPattern",
    "ConstantPattern",
    "DTypePattern",
    "FuncParserContext",
    "FunctionPattern",
    "FunctionRole",
    "LayoutPattern",
    "LexicalScope",
    "MatchContext",
    "ModuleBuildContext",
    "NamePattern",
    "ParseError",
    "PatternFailure",
    "RenderVisitor",
    "ScalarTypePattern",
    "ShapePattern",
    "SignaturePattern",
    "StatementPattern",
    "StoragePattern",
    "TensorOptionalSlotPattern",
    "TensorPattern",
    "TypeAnnotationPattern",
    "consume_module_context",
    "create_module_context",
    "module_context_for_frame",
    "parse_node",
    "register_module_context",
    "render_grammar",
]


def _constant(value):
    if isinstance(value, bool):
        dtype = runtime.DType.bool
    elif isinstance(value, int):
        dtype = runtime.DType.i64
    elif isinstance(value, float):
        dtype = runtime.DType.f32
    else:
        raise TypeError(type(value).__name__)
    return runtime.Constant(
        type=runtime.TensorType.scalar(dtype, storage=runtime.StorageKind.UMAT),
        value=value,
    )


def _infer_call(operation, args, context):
    placeholder_type = getattr(operation, "type", None)
    if placeholder_type is None and args:
        placeholder_type = args[0].type
    if placeholder_type is None:
        placeholder_type = runtime.TensorType.scalar(runtime.DType.f32)
    metadata = ()
    if context.function is not None and context.function.state.mesh_stack:
        metadata = (
            runtime.ExecutionDomainMetadata(tuple(context.function.state.mesh_stack)),
        )
    placeholder = runtime.Call(
        type=placeholder_type, target=operation, args=tuple(args), metadata=metadata
    )
    infer_context = context.lexical_scope.lookup(_TYPE_INFER_CONTEXT)
    if not isinstance(infer_context, runtime.TypeInferContext):
        infer_context = runtime.TypeInferContext()
    inferred = runtime.TypeInferVisitor(infer_context).visit(placeholder)
    return dataclasses.replace(placeholder, type=inferred)


def _slice_size(begin, end, stride, context, node):
    try:
        value = runtime.normalize_dim(
            runtime.slice_size(
                runtime.dim_expr(begin),
                runtime.dim_expr(end),
                runtime.dim_expr(stride),
            )
        )
    except (TypeError, ValueError) as error:
        raise ParseError.from_node(node, context, str(error)) from error
    if isinstance(value, runtime.Constant):
        return value.value
    return value


from .grammar_render import RenderVisitor, render_grammar
from .pattern_nodes import *
