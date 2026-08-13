from __future__ import annotations

import ast
import dataclasses
import enum
import inspect
import logging
import textwrap
from typing import Any, Callable

from tilefoundry.ir.core import (
    BindingMetadata,
    Call,
    Constant,
    Expr,
    IRMetadata,
    SourceSpanMetadata,
    Tuple,
    TypeInferContext,
    Var,
    VerifyError,
    get_metadata,
    replace_metadata,
)
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.op_schema import OpSchema
from tilefoundry.ir.hir._call_binding import bound_params, set_authoring_reader
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.function import elaborate
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.ir.types.dim import DimAdd, is_dim_expr, simplify_dim
from tilefoundry.ir.types.dtype import FloatDType
from tilefoundry.ir.types.shape_helpers import i64_const
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, shard_layout_of
from tilefoundry.ir.types.storage import StorageKind, resolve_storage
from tilefoundry.visitor_registry import typeinfer_registry

from .dispatch import (
    Token,
    _binary_kind_for_ast_op,
    _unary_kind_for_ast_op,
    resolve_callable,
    resolve_op,
    resolve_schema,
    resolve_stmt,
)
from .static_eval import eval_static
from .sugar import (
    LayoutSugarError,
    _is_tuple_sugar,
    parse_layout_sugar,
    parse_shard_layout_sugar,
    try_parse_sugar_tensor_type,
)
from .symtab import LexicalEnv

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _ModuleCallee(IRMetadata):
    """The class-body binding a call reached its callee's Module through.

    Authoring state private to the parser: a class body is parsed before its
    children are attached, and attaching one copies it, so the binding name is
    what says which attached child the call meant. ``@module`` collection
    rebuilds against that child and takes this record off.
    """

    binding: str
    owner: Module


def _authored_child_call(call):
    """The child Module *call* was written through, else ``None``.

    The authoring phase's answer to which calls carry activations only. It holds
    until ``@module`` collection consumes the record, and asks nothing of the
    walk: the record is on the call site itself.
    """
    record = get_metadata(call, _ModuleCallee)
    return None if record is None else record.owner


set_authoring_reader(_authored_child_call)


_IR_OBJECT_TYPES = {
    "Topology": None,
    "Mesh": None,
    "ShardLayout": None,
    "Layout": None,
}


def _warn_if_ir_object(val: Any, name: str) -> None:
    """Warn when a preconstructed IR object is resolved from closure.

    Canonical DSL source should use AST constructor syntax instead of capturing
    prebuilt Python objects in the closure.
    """
    type_name = type(val).__name__
    if type_name in _IR_OBJECT_TYPES:
        logger.warning(
            "Closure-captured IR object %r of type %s — "
            "this is not canonical. Prefer declaring in DSL source or "
            "using topology-name string resolution.",
            name, type_name,
        )


def extract_ast(fn) -> ast.FunctionDef:
    source_lines, start_line = inspect.getsourcelines(fn)
    src = textwrap.dedent("".join(source_lines))
    mod = ast.parse(src)
    ast.increment_lineno(mod, start_line - 1)


    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef):
            return node
    raise VerifyError("cannot locate FunctionDef in source")


def _collect_closure(fn, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect a live Python function's name-resolution namespace.

    Shared by ``parse_func`` (HIR) and ``parse_prim_func`` (TIR): ``extra``
    (sibling ``@func`` / ``@prim_func`` bindings from a ``@module`` class
    body's definition frame) sits below the function's own globals /
    freevars so it cannot shadow them.
    """
    closure: dict[str, Any] = {}
    if extra:
        closure.update(extra)
    if fn.__globals__ is not None:
        closure.update(fn.__globals__)
    if fn.__closure__ is not None:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            try:
                closure[name] = cell.cell_contents
            except ValueError:
                pass
    return closure


def _annotation_head_name(node: ast.AST) -> str | None:
    """Annotation head name.

    Return the subscript base identifier (``Tensor`` / ``ConstTensor``),
    resolving through an attribute path such as ``dsl.ConstTensor``.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_const_tensor_annotation(node: ast.AST) -> bool:
    """``ConstTensor[...]`` marks a parameter ``is_const=True``.

    ``ConstTensor[...]`` marks a parameter ``is_const=True``; ``Tensor[...]``
    and every other annotation form leave it ``False``.
    """
    return (
        isinstance(node, ast.Subscript)
        and _annotation_head_name(node.value) == "ConstTensor"
    )


def _resolve_tensor_type(node: ast.AST, closure: dict[str, Any]) -> TensorType:
    """Resolve tensor annotations identically for HIR and TIR functions.

    Parse compact layout sugar directly from AST or evaluate the verbose shard
    layout form in *closure*. See
    [parser §1.4](docs/spec/parser.md#14-tensor-and-consttensor-annotations) and
    [parser §1.5](docs/spec/parser.md#15-layout-sugar).
    """
    result = try_parse_sugar_tensor_type(node, closure)
    if result is not None:
        return result
    try:
        code = compile(ast.Expression(body=node), "<ann>", "eval")
        val = eval(code, closure)  # noqa: S307 — controlled internal eval
    except Exception as exc:
        raise VerifyError(f"failed to resolve type annotation: {exc}")
    if isinstance(val, TensorType):
        return val
    raise VerifyError(f"annotation did not resolve to TensorType, got {type(val).__name__}")


def _build_params(
    node: ast.FunctionDef,
    closure: dict[str, Any],
    resolve_annotation: Callable[[ast.AST, dict[str, Any]], TensorType],
    *,
    decorator_name: str,
) -> tuple[Var, ...]:
    """Build ``Var`` parameters from a function's AST arg annotations.

    Shared by ``parse_func`` (HIR) and ``parse_prim_func`` (TIR); both pass
    :func:`_resolve_tensor_type` as *resolve_annotation* so a ``Tensor[...]``
    layout-sugar annotation works identically on ``@func`` and ``@prim_func``
    params.
    """
    out: list[Var] = []
    for a in node.args.args:
        if a.annotation is None:
            raise VerifyError(f"{decorator_name} param {a.arg!r} must be annotated")
        ann_type = resolve_annotation(a.annotation, closure)
        is_const = _is_const_tensor_annotation(a.annotation)
        out.append(Var(type=ann_type, name=a.arg, is_const=is_const))
    return tuple(out)


def _i64(value: int) -> Constant:
    return i64_const(value)


def _constant_from_py(value: Any) -> Constant:


    if isinstance(value, bool):
        return Constant(type=TensorType.scalar(DType.bool, storage=StorageKind.UMAT), value=value)
    if isinstance(value, int):
        return Constant(type=TensorType.scalar(DType.i64, storage=StorageKind.UMAT), value=value)
    if isinstance(value, float):
        return Constant(type=TensorType.scalar(DType.f32, storage=StorageKind.UMAT), value=value)
    raise VerifyError(f"unsupported literal type {type(value).__name__}")




_STATIC_ARITH_NODES: tuple[type, ...] = (
    ast.Constant, ast.Name, ast.Attribute, ast.UnaryOp, ast.BinOp,
)


def _is_python_float_scalar(expr: Expr) -> bool:
    """Whether *expr* is a Python float scalar, which carries no precision."""
    ty = expr.type
    return (
        isinstance(expr, Constant)
        and isinstance(ty, TensorType)
        and ty.shape == ()
        and ty.storage is StorageKind.UMAT
        and isinstance(ty.dtype, FloatDType)
    )


def _with_python_float_dtypes(args: tuple[Expr, ...]) -> tuple[Expr, ...]:
    """Give each Python float scalar the float dtype its fellow operands carry.

    Applies to floats only; a Python integer keeps its own dtype.
    """
    floats = {i for i, arg in enumerate(args) if _is_python_float_scalar(arg)}
    if not floats:
        return args
    others = {
        arg.type.dtype
        for i, arg in enumerate(args)
        if i not in floats
        and isinstance(arg.type, TensorType)
        and isinstance(arg.type.dtype, FloatDType)
    }
    if len(others) != 1:
        return args
    dtype = next(iter(others))
    out = list(args)
    for i in floats:
        if out[i].type.dtype != dtype:
            out[i] = dataclasses.replace(
                out[i], type=dataclasses.replace(out[i].type, dtype=dtype)
            )
    return tuple(out)


class BaseExprVisitor:
    """Shared visitor for Expr-returning AST nodes. Emits core_ir Expr."""

    token: Token

    resolves_module_callees = False

    def __init__(
        self, env: LexicalEnv, closure: dict[str, Any], *, in_module_body: bool = False
    ):
        self.env = env
        self.closure = closure
        self.in_module_body = in_module_body



        self._ctx = TypeInferContext()



        self._explicit_binding_call_ids: set[int] = set()




        self._call_dsl_names: dict[int, str] = {}
        self._scalar_index_ids: set[int] = set()



        self._tile_windows: dict[int, tuple[Any, Any]] = {}
        self._active_source_node: ast.AST | None = None
        self._active_binding_hint: str | None = None
        self.source_filename = "<string>"

    def _tuple_expr_expr(self, node: ast.Tuple):
        """Build a ``Tuple`` from an AST tuple literal."""
        elements = tuple(self.expr(e) for e in node.elts)
        field_types = tuple(e.type for e in elements)
        return Tuple(type=TupleType(fields=field_types), elements=elements)

    def _resolve_body_mesh(self, name: str):
        """Resolve a mesh by variable name from the lexical env only.

        Body sugar (``reshard(layout=(... @ mesh.axis, ...))``) must use
        meshes from lexical ``with Mesh(...) as name`` scopes.  Closure /
        global mesh IR objects are NOT accepted for body sugar.
        """
        val = self.env.lookup(name)
        if isinstance(val, Mesh):
            return val
        return None

    def _current_default_mesh(self):
        """Return the innermost Mesh from the lexical scope, or None.

        Used as the *default_mesh* for all-Broadcast ShardLayout sugar.
        """
        return self.env.innermost_mesh()

    def _contains_mesh_coordinate(self, node: ast.AST) -> bool:
        """Whether *node* contains a dialect-specific mesh coordinate."""
        return False



    def expr(self, node: ast.AST) -> Expr:
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise VerifyError(f"unsupported AST node in expression: {type(node).__name__}")
        previous = self._active_source_node
        self._active_source_node = node
        try:
            return method(node)
        finally:
            self._active_source_node = previous

    def expr_with_binding(self, node: ast.AST, name: str) -> Expr:
        """Parse one RHS while making its authored LHS available to errors."""
        previous = self._active_binding_hint
        self._active_binding_hint = name
        try:
            return self.expr(node)
        finally:
            self._active_binding_hint = previous

    @staticmethod
    def _attach_metadata(expr: Expr, value: IRMetadata) -> None:
        """Attach parser-authored metadata without rebuilding the SSA node."""
        kept = tuple(item for item in expr.metadata if type(item) is not type(value))
        object.__setattr__(expr, "metadata", (*kept, value))

    def _source_span(self) -> SourceSpanMetadata | None:
        node = self._active_source_node
        if node is None or not hasattr(node, "lineno"):
            return None
        return SourceSpanMetadata(
            file=self.source_filename,
            line=node.lineno,

            column=node.col_offset + 1,
            end_line=getattr(node, "end_lineno", None),
            end_column=(
                getattr(node, "end_col_offset", None) + 1
                if getattr(node, "end_col_offset", None) is not None
                else None
            ),
        )

    def _source_metadata(self) -> tuple:
        metadata = []
        span = self._source_span()
        if span is not None:
            metadata.append(span)
        if self._active_binding_hint is not None:
            metadata.append(BindingMetadata(self._active_binding_hint))
        return tuple(metadata)

    @staticmethod
    def _with_binding(expr: Expr, name: str) -> Expr:
        return replace_metadata(expr, BindingMetadata(name))



    def visit_Constant(self, node: ast.Constant) -> Expr:
        return self._constant_expr(node.value)

    def _constant_expr(self, value: Any) -> Expr:
        constant = _constant_from_py(value)
        span = self._source_span()
        return replace_metadata(constant, span) if span is not None else constant

    def _static_number(self, node: ast.AST):
        """The number *node* already is, or ``None`` to leave it to the IR path."""
        try:
            value = eval_static(
                node,
                closure=self.closure,
                lookup=self.env.lookup,
                allowed_nodes=_STATIC_ARITH_NODES,
                attr_resolver=self._resolve_static_attribute,
            )
        except VerifyError:
            return None
        return value if isinstance(value, (int, float)) else None

    def _static_iterable(self, node: ast.AST):
        """The compile-time sequence a comprehension walks.

        The compile-time sequence a comprehension walks: builtin ``range`` over
        compile-time integers, or a tuple / list of compile-time values.

        No other call is evaluated here — resolving one would run it.
        """
        if isinstance(node, ast.Call):
            shadowed = self.env.lookup("range") is not None or "range" in self.closure
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id != "range"
                or shadowed
                or node.keywords
            ):
                raise VerifyError(
                    f"compile-time list comprehension iterates `range(...)` or a "
                    f"compile-time sequence, not {ast.unparse(node)!r}"
                )
            bounds = [self._static_number(arg) for arg in node.args]
            if not bounds or any(
                isinstance(bound, bool) or not isinstance(bound, int) for bound in bounds
            ):
                raise VerifyError("`range(...)` here takes compile-time integers")
            return range(*bounds)
        value = eval_static(
            node,
            closure=self.closure,
            lookup=self.env.lookup,
            allowed_nodes=(*_STATIC_ARITH_NODES, ast.Tuple, ast.List),
            attr_resolver=self._resolve_static_attribute,
        )
        if not isinstance(value, (tuple, list, range)):
            raise VerifyError(
                f"compile-time list comprehension iterates a compile-time sequence, "
                f"got {type(value).__name__}"
            )
        return value

    def _static_expr_list(self, node: ast.AST) -> "list[Expr] | None":
        """A compile-time list of IR expressions, or ``None`` when *node* is not one.

        The list stays Python; only its elements are Exprs.
        """
        if isinstance(node, ast.List):
            return [self.expr(el) for el in node.elts]
        if not isinstance(node, ast.ListComp):
            return None
        if len(node.generators) != 1:
            raise VerifyError("compile-time list comprehension takes one `for` clause")
        generator = node.generators[0]
        if generator.ifs or generator.is_async:
            raise VerifyError(
                "compile-time list comprehension takes no `if` guard and is not async"
            )
        if not isinstance(generator.target, ast.Name):
            raise VerifyError("compile-time list comprehension binds one plain name")
        values = list(self._static_iterable(generator.iter))
        self.env.push_frame()
        try:
            items = []
            for value in values:
                self.env.define(generator.target.id, value)
                items.append(self.expr(node.elt))
        finally:
            self.env.pop_frame()
        return items



    def visit_Name(self, node: ast.Name) -> Expr:
        val = self.env.lookup(node.id)
        from_closure = False
        if val is None:
            val = self.closure.get(node.id)
            from_closure = True
        if val is None:
            raise VerifyError(f"undefined name {node.id!r}")
        if isinstance(val, Expr):
            return val
        if isinstance(val, slice):
            return val.start
        if isinstance(val, (int, float, bool)):
            return _constant_from_py(val)


        if from_closure and type(val).__name__ in _IR_OBJECT_TYPES:
            _warn_if_ir_object(val, node.id)
        raise VerifyError(f"name {node.id!r} resolved to non-Expr Python value {type(val).__name__}")



    def visit_Attribute(self, node: ast.Attribute) -> Expr:

        value = self._static_number(node)
        if value is not None:
            return self._constant_expr(value)

        raise VerifyError(f"attribute access {ast.unparse(node)!r} not valid as Expr")



    def visit_Subscript(self, node: ast.Subscript) -> Expr:
        """Resolve ``expr[idx]`` to a ``TupleGetItem`` or ``Slice`` Call.

        - a compile-time list + integer index → the element it holds.
        - ``TupleType`` value + int constant index → ``TupleGetItem``.
        - ``TensorType`` + slices / tile-window Names → ``Slice``.
        - Compile-time integers and scalar range induction Names also reshape
          away their selected axis, matching torch indexing.
        """
        if isinstance(node.value, ast.Name):
            bound = self.env.lookup(node.value.id)
            if isinstance(bound, list):
                return self._list_element(node.value.id, bound, node.slice)
        value = self.expr(node.value)
        if isinstance(value.type, TupleType):
            slc = node.slice
            if not (isinstance(slc, ast.Constant) and isinstance(slc.value, int)
                    and not isinstance(slc.value, bool)):
                raise VerifyError(
                    "subscript on TupleType requires an integer constant index"
                )
            return self._build_call(TupleGetItem(index=slc.value), (value,))
        if isinstance(value.type, TensorType):
            return self._lift_tensor_subscript(value, node.slice)
        raise VerifyError(
            f"subscript only supported on TupleType / TensorType (got "
            f"{type(value.type).__name__})"
        )

    def _lift_tensor_subscript(self, value, slc: ast.AST):
        """Lift ``x[slice0, slice1, ...]`` to a ``Slice`` Op call.

        Each subscript element is one of:
        - ``ast.Slice`` — full or partial ``start:stop[:step]``;
        - an ``ast.Name`` resolving to a Python ``slice`` parser-side
          binding (``for ok in tile(extent, step)``).

        Other forms (constants, computed Expr indices, ellipsis, lists)
        are deferred to indexed read/write ops and raise here.
        """
        if isinstance(slc, ast.Tuple):
            elts = list(slc.elts)
        else:
            elts = [slc]

        x_ty = value.type
        if not isinstance(x_ty, TensorType):  # pragma: no cover — guarded above
            raise VerifyError("tensor subscript: value must be TensorType")
        if len(elts) != len(x_ty.shape):
            raise VerifyError(
                f"tensor subscript rank {len(elts)} != tensor rank "
                f"{len(x_ty.shape)}"
            )
        if shard_layout_of(x_ty.layout) is not None and any(
            self._contains_mesh_coordinate(el) for el in elts
        ):
            raise VerifyError(
                "tensor subscript uses a mesh coordinate to index an already "
                "placed tensor; data-dependent mesh ownership is unresolved"
            )

        starts: list[Expr] = []
        sizes: list[Any] = []
        strides: list[Any] = []
        collapsed: list[int] = []
        for axis, (el, dim) in enumerate(zip(elts, x_ty.shape)):
            index = self._integer_index(el, dim)
            if index is not None:
                starts.append(i64_const(index))
                sizes.append(1)
                strides.append(1)
                collapsed.append(axis)
                continue
            scalar_index = self._scalar_index(el)
            if scalar_index is not None:
                starts.append(scalar_index)
                sizes.append(1)
                strides.append(1)
                collapsed.append(axis)
                continue
            b, e, s = self._slicer_for_dim(el, dim, axis)
            b_expr = b if isinstance(b, Expr) else i64_const(int(b))
            e_expr = e if isinstance(e, Expr) else i64_const(int(e))
            s_expr = s if isinstance(s, Expr) else i64_const(int(s))
            starts.append(b_expr)
            from tilefoundry.ir.hir.tensor.slice import slice_size  # noqa: PLC0415

            size = slice_size(b_expr, e_expr, s_expr)
            if not is_dim_expr(size):
                raise VerifyError(
                    f"tensor subscript axis {axis}: a run-time start needs the "
                    "stop endpoint written as `start + K` for a compile-time K, "
                    "because Slice takes a static size"
                )
            sizes.append(int(size.value) if isinstance(size, Constant) else size)
            strides.append(s)

        starts_expr = Tuple(
            type=TupleType(fields=tuple(start.type for start in starts)),
            elements=tuple(starts),
        )
        sliced = self._build_call(
            Slice(sizes=tuple(sizes), strides=tuple(strides)),
            (value, starts_expr),
        )
        if not collapsed:
            return sliced
        kept = tuple(
            dim for axis, dim in enumerate(sliced.type.shape) if axis not in collapsed
        )
        return self._build_call(Reshape(new_shape=kept), (sliced,))

    def _integer_index(self, el: ast.AST, dim: Any) -> "int | None":
        """The compile-time integer this element is, counted from the front.

        ``ast.Slice`` and a tile-window ``slice`` name keep their axis and are left
        to ``_slicer_for_dim``.
        """
        if isinstance(el, ast.Slice):
            return None
        if isinstance(el, ast.Name) and isinstance(self.env.lookup(el.id), slice):
            return None
        value = self._static_number(el)
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        if value < 0 or isinstance(dim, int):
            if not isinstance(dim, int):
                raise VerifyError(
                    f"tensor subscript index {value}: counting back from the end needs "
                    f"a static extent, and this axis is {dim}"
                )
            normalized = value + dim if value < 0 else value
            if not 0 <= normalized < dim:
                raise VerifyError(
                    f"tensor subscript index {value} is out of range for extent {dim}"
                )
            return normalized
        return value

    def _scalar_index(self, el: ast.AST) -> "Expr | None":
        """A registered scalar induction index from a ``range`` loop."""
        if not isinstance(el, ast.Name):
            return None
        value = self.env.lookup(el.id)
        type_ = value.type if isinstance(value, Expr) else None
        if (
            id(value) in self._scalar_index_ids
            and isinstance(type_, TensorType)
            and type_.shape == ()
            and type_.dtype is DType.i64
        ):
            return value
        return None

    def _list_element(self, name: str, items: list, slc: ast.AST) -> Expr:
        """One element of a compile-time list, taken by compile-time index."""
        index = self._static_number(slc)
        if isinstance(index, bool) or not isinstance(index, int):
            raise VerifyError(
                f"{name!r} is a compile-time list, so its index must be a "
                f"compile-time integer"
            )
        if not -len(items) <= index < len(items):
            raise VerifyError(
                f"{name!r} holds {len(items)} entries; index {index} is out of range"
            )
        return items[index]

    def _slicer_for_dim(self, el: ast.AST, dim: Any, axis: int):
        """Resolve one subscript element to ``(begin, end, stride)``.

        ``dim`` is the input tensor's static shape value at this axis
        (used as the default upper bound for ``:``).
        """
        if isinstance(el, ast.Slice):

            if el.lower is None:
                begin = 0
            else:
                begin = self._slicer_endpoint(el.lower)
            if el.upper is None:
                end = dim
            else:
                end = self._slicer_endpoint(el.upper)
            if el.step is None:
                stride = 1
            else:
                stride = self._eval_static(el.step)
            if not is_dim_expr(stride):
                raise VerifyError(
                    f"tensor subscript axis {axis}: slice stride must be a "
                    "compile-time dimension"
                )
            if all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (dim, begin, end, stride)
            ) and stride > 0:
                begin, end, stride = slice(begin, end, stride).indices(dim)
            return begin, end, stride
        if isinstance(el, ast.Name):
            val = self.env.lookup(el.id)
            if isinstance(val, slice):
                return val.start, val.stop, val.step
        moved = self._moved_tile_window(el, dim, axis)
        if moved is not None:
            return moved.start, moved.stop, moved.step
        raise VerifyError(
            f"tensor subscript axis {axis}: unsupported indexer "
            f"{ast.dump(el)} (expected `:`, `a:b`, a tile-window slice, or a "
            f"tile window moved by a compile-time integer)"
        )

    def _slicer_endpoint(self, node: ast.AST):
        """Resolve a slice endpoint, admitting runtime scalar dim arithmetic."""
        try:
            return self._eval_static(node, allow_runtime_scalar=True)
        except VerifyError:
            return self.expr(node)

    def _tile_window(self, node: ast.AST) -> "slice | None":
        """The tile window *node* names, else ``None``."""
        if not isinstance(node, ast.Name):
            return None
        value = self.env.lookup(node.id)
        if isinstance(value, slice) and id(value.start) in self._tile_windows:
            return value
        return None

    def _window_move(self, el: ast.AST) -> "tuple[slice, int] | None":
        """The tile window *el* reads and the offset that moves it.

        ``None`` when *el* names no window. Offsets accumulate, so
        ``i + QN + KN`` and ``QN + KN + i`` name one move by one sum, and each
        term is a compile-time integer on its own.
        """
        window = self._tile_window(el)
        if window is not None:
            return window, 0
        if not isinstance(el, ast.BinOp) or not isinstance(el.op, (ast.Add, ast.Sub)):
            return None
        sign = -1 if isinstance(el.op, ast.Sub) else 1
        moved = self._window_move(el.left)
        offset_node = el.right
        if moved is None:
            moved = self._window_move(el.right)
            if moved is None:
                return None
            if sign == -1:
                raise VerifyError(
                    f"{ast.unparse(el)!r}: an offset moves a window, so the window "
                    f"is what the offset is added to -- subtracting it from "
                    f"{ast.unparse(el.left)!r} reverses the window rather than "
                    f"moving it"
                )
            offset_node = el.left
        offset = self._static_number(offset_node)
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise VerifyError(
                f"{ast.unparse(el)!r}: a tile window moves by a compile-time "
                f"integer, and {ast.unparse(offset_node)!r} is not one"
            )
        window, carried = moved
        return window, carried + sign * offset

    def _moved_tile_window(self, el: ast.AST, dim: Any, axis: int) -> "slice | None":
        """The window ``i + C`` reads, or ``None`` when *el* names no window.

        A tile window is a length bound to a moving base, so a compile-time
        offset moves the base and leaves the length alone: ``i + C`` reads
        ``[lo + C, lo + C + step)``. The base was already computed at compile
        time, so this axis keeps the static extent ``i`` alone gives it.
        """
        move = self._window_move(el)
        if move is None:
            return None
        window, offset = move
        if offset == 0:
            return window
        extent, length = self._tile_windows[id(window.start)]
        self._check_moved_window(el, axis, offset, extent, length, dim)
        base = simplify_dim(DimAdd, (window.start, offset))
        return slice(base, simplify_dim(DimAdd, (base, length)), window.step)

    @staticmethod
    def _check_moved_window(
        el: ast.AST, axis: int, offset: int, extent: Any, length: Any, dim: Any
    ) -> None:
        """Refuse a move that reads off the axis.

        The loop domain, the window length, the offset and the axis extent are
        all compile-time, so the last window a moved read touches is too. A
        symbolic extent leaves the bound to evaluate time, which is where an
        unmoved window's own tail is caught.
        """
        if offset < 0:
            raise VerifyError(
                f"tensor subscript axis {axis}: {ast.unparse(el)!r} moves the "
                f"window back by {-offset}, and this loop's first window starts "
                f"at 0, so it would begin before the axis"
            )
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (extent, length, dim)
        ):
            return
        if extent <= 0 or length <= 0:
            return
        last = (extent - 1) // length * length
        if last + offset + length > dim:
            raise VerifyError(
                f"tensor subscript axis {axis}: {ast.unparse(el)!r} reads "
                f"[{last + offset}, {last + offset + length}) on its last "
                f"iteration, and the axis is {dim} long"
            )



    def visit_BinOp(self, node: ast.BinOp) -> Expr:

        value = self._static_number(node)
        if value is not None:
            return self._constant_expr(value)
        opname = type(node.op).__name__

        if opname == "MatMult":
            matmul_cls = resolve_op("matmul")
            if matmul_cls is None:
                raise VerifyError("matmul op not registered")
            left = self.expr(node.left)
            right = self.expr(node.right)
            return self._build_call(matmul_cls(), (left, right))
        kind = _binary_kind_for_ast_op(opname)
        if kind is None:
            raise VerifyError(f"unsupported binary op {opname}")
        left = self.expr(node.left)
        right = self.expr(node.right)
        return self._build_call(self._make_binary(kind), (left, right))



    def visit_Compare(self, node: ast.Compare) -> Expr:
        if len(node.ops) != 1:
            raise VerifyError("chained comparison not supported in V1")
        opname = type(node.ops[0]).__name__
        kind = _binary_kind_for_ast_op(opname)
        if kind is None:
            raise VerifyError(f"unsupported comparison {opname}")
        left = self.expr(node.left)
        right = self.expr(node.comparators[0])
        return self._build_call(self._make_binary(kind), (left, right))

    def visit_BoolOp(self, node: ast.BoolOp) -> Expr:
        opname = type(node.op).__name__
        kind = _binary_kind_for_ast_op(opname)
        if kind is None:
            raise VerifyError(f"unsupported bool op {opname}")
        if len(node.values) != 2:
            raise VerifyError("bool op requires exactly 2 operands in V1")
        left = self.expr(node.values[0])
        right = self.expr(node.values[1])
        return self._build_call(self._make_binary(kind), (left, right))

    @staticmethod
    def _make_binary(kind):
        return Binary(kind=kind)

    @staticmethod
    def _make_unary(kind):
        return Unary(kind=kind)



    def visit_UnaryOp(self, node: ast.UnaryOp) -> Expr:
        opname = type(node.op).__name__
        kind = _unary_kind_for_ast_op(opname)
        if kind is None:
            raise VerifyError(f"unsupported unary op {opname}")
        operand = self.expr(node.operand)
        return self._build_call(self._make_unary(kind), (operand,))

    def visit_Call(self, node: ast.Call) -> Expr:
        return self.call_to_op_call(node)



    def _resolve_call_target(self, func: ast.AST):
        """Resolve a bare or namespaced callee to an operation schema.

        Lexical and closure bindings precede dialect-strict registry lookup.
        Return schemas so aliases and operation classes share one builder path;
        match namespace modules by identity. See
        [parser §4.2](docs/spec/parser.md#42-closure-then-registry-callee-resolution)
        and [parser §4.6](docs/spec/parser.md#46-per-dialect-strict-resolution).
        """
        if isinstance(func, ast.Name):
            val = self.env.lookup(func.id)
            if val is None:
                val = self.closure.get(func.id)
            schema = self._schema_from_value(val)
            if schema is not None:
                return schema
            try:
                _kind, cls = resolve_callable(func.id, self.token)
            except VerifyError:
                return None
            return getattr(cls, "_op_schema", None)
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
        ):
            ns = self.env.lookup(func.value.id)
            if ns is None:
                ns = self.closure.get(func.value.id)
            if ns is None:
                return None


            # noqa cycle: tilefoundry.dsl pulls tilefoundry.parser.overload, which

            import tilefoundry.dsl as _dsl  # noqa: PLC0415
            if ns is _dsl.tf:
                return resolve_schema(func.attr, "tf")
            if ns is _dsl.T:
                return resolve_schema(func.attr, "T")
        return None

    def _resolve_function_target(self, func: ast.AST):
        """Resolve function target.

        Return ``(hir.Function, binding, owner)`` behind a callee AST, or
        ``(None, None, None)`` when the callee is not an ``@func``-decorated function.
        ``@func`` evaluates to the ``hir.Function`` directly, so a sibling
        callee binding *is* that Function (see :func:`tilefoundry.script.func`).
        A bare name bound to a ``Module`` is that Module's entry function and
        reports the name as *binding* and that Module as *owner*; the attribute
        spelling is refused.
        """
        val: Any = None
        if isinstance(func, ast.Name):
            val = self.env.lookup(func.id)
            if val is None:
                val = self.closure.get(func.id)
            if isinstance(val, Module) and self.resolves_module_callees:
                if not self.in_module_body:
                    raise VerifyError(
                        f"{func.id!r}: a Module is called only from a function "
                        f"authored in a @module class body, which is what attaches "
                        f"{val.name!r} as a child and gives the call a child to "
                        f"reach; this function declares none"
                    )
                return self._module_entry_target(val, func.id), func.id, val
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            owner = self.env.lookup(func.value.id)
            if owner is None:
                owner = self.closure.get(func.value.id)
            if isinstance(owner, Module) and self.resolves_module_callees:
                declares = (
                    f"entry {owner.entry!r}" if owner.entry is not None else "no entry"
                )
                raise VerifyError(
                    f"{ast.unparse(func)!r}: a Module is called through its bare "
                    f"binding, which calls its entry function; Module {owner.name!r} "
                    f"declares {declares}, so write {func.value.id}(...) rather than "
                    f"reaching in for {func.attr!r}"
                )
        if isinstance(val, HirFunction):
            return val, None, None
        return None, None, None

    @staticmethod
    def _module_entry_target(owner: Module, name: str) -> HirFunction:
        """The hir Function *owner* is entered through.

        A Module that cannot answer is refused here, naming what it could not
        answer, rather than falling through to *unknown Op name*.
        """
        try:
            entry = owner.entry_function()
        except ValueError as exc:
            raise VerifyError(
                f"{name!r}: calling a Module calls its entry function -- {exc}"
            ) from exc
        if not isinstance(entry, HirFunction):
            raise VerifyError(
                f"{name!r}: calling a Module calls its entry function, and Module "
                f"{owner.name!r} enters through {entry.name!r}, a "
                f"{type(entry).__name__} rather than an hir Function"
            )
        return entry

    def _build_function_call(
        self, callee: Any, node: ast.Call, name: str,
        module_binding: str | None = None, owner: Module | None = None,
    ) -> Expr:
        """Build a nested HIR function call with an elaborated target.

        Enforce arity before binding argument types so the call targets its
        per-site function instance. Accept only positional IR arguments plus the
        explicit ``loc=`` binding label. See
        [hir §1.1](docs/spec/hir.md#11-function).
        """
        explicit_loc: str | None = None
        explicit_loc_given = False
        extra_kwargs: list[str] = []
        for k in node.keywords:
            if k.arg == "loc":
                explicit_loc = self._eval_static(k.value)
                explicit_loc_given = True
                continue
            extra_kwargs.append(k.arg)
        if extra_kwargs:
            raise VerifyError(
                f"{name!r}: nested @func call does not accept keyword args "
                f"{extra_kwargs!r} (positional-only at the IR level)"
            )
        module_call = module_binding is not None
        expected = len(bound_params(callee, from_reading=module_call))
        got = len(node.args)
        if got != expected and module_call:
            raise VerifyError(
                f"{name!r}: Module {name!r} takes {expected} activation(s) — its "
                f"{len(callee.params) - expected} ConstTensor parameter(s) come from "
                f"that Module's own bindings — but got {got}"
            )
        if got != expected:
            raise VerifyError(
                f"{name!r}: nested @func call arity mismatch — callee "
                f"declares {expected} parameter(s), call passed {got}"
            )
        input_args = tuple(self.expr(a) for a in node.args)
        records = () if owner is None else (_ModuleCallee(module_binding, owner),)
        call_for_errors = Call(
            type=callee.return_type, target=callee, args=input_args,
            metadata=(*self._source_metadata(), *records),
        )
        if explicit_loc_given:
            call_for_errors = replace_metadata(
                call_for_errors, BindingMetadata(explicit_loc)
            )
        instance = elaborate(
            callee, tuple(a.type for a in input_args), self._ctx,
            call=call_for_errors,
        )
        call = self._build_call(instance, input_args, records=records)
        if explicit_loc_given:
            call = replace_metadata(call, BindingMetadata(explicit_loc))
            self._explicit_binding_call_ids.add(id(call))

        self._call_dsl_names[id(call)] = name
        return call

    @staticmethod
    def _schema_from_value(val):
        """Extract an ``OpSchema`` from a bound DSL surface value.

        Accepts:
        - an ``OpSchema`` instance directly;
        - any object carrying an ``_op_schema`` attribute (Op class
          set by ``@register_op``; alias builder fn set by
          ``@register_alias``).
        Returns ``None`` for anything else.
        """
        if isinstance(val, OpSchema):
            return val
        schema = getattr(val, "_op_schema", None)
        if isinstance(schema, OpSchema):
            return schema
        return None

    def call_to_op_call(self, node: ast.Call) -> Expr:
        """Resolve ``foo(...)`` to a ``Call`` on an hir Op.

        Dispatches a callee that is an ``hir.Function`` (nested ``@func`` call)
        to :meth:`_build_function_call`; otherwise resolves an ``OpSchema`` via
        :meth:`_resolve_call_target` and binds positional / keyword args to the
        schema's input / attribute ParamDefs. Raises when the callee is a tir
        Stmt op (the caller handles Stmt position) or is unresolved.
        """
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = ast.unparse(node.func)
        else:
            raise VerifyError("only Name / Attribute callees supported in V1")





        callee_func, module_binding, owner = self._resolve_function_target(node.func)
        if callee_func is not None:
            return self._build_function_call(
                callee_func, node, name, module_binding, owner
            )

        schema = self._resolve_call_target(node.func)
        if schema is None:

            if isinstance(node.func, ast.Name) and resolve_stmt(name) is not None:
                raise VerifyError(
                    f"{name!r} is an effect Stmt op; cannot appear in Expr position "
                    f"(wrap in Assign or use as top-level Stmt)"
                )
            raise VerifyError(f"unknown Op name {name!r}")


        param_infos = schema.signature
        input_params = [p for p in param_infos if p.kind == "input"]
        attr_params = [p for p in param_infos if p.kind == "attribute"]








        is_variadic = bool(getattr(getattr(schema, "op_class", None), "is_variadic", False))


        pos_args = list(node.args)
        input_args = []
        attr_kwargs: dict[str, Any] = {}

        if is_variadic:
            if len(input_params) != 1:
                raise VerifyError(
                    f"{name!r}: variadic op schema must declare exactly one "
                    f"input ParamDef, got {len(input_params)}"
                )
            for arg in pos_args:
                input_args.append(self.expr(arg))
        else:
            for i, arg in enumerate(pos_args):
                if i < len(input_params):
                    if (
                        isinstance(arg, ast.Tuple)
                        and (
                            (
                                schema.name == "insert_slice"
                                and input_params[i].name == "offsets"
                            )
                            or (
                                schema.name == "slice"
                                and input_params[i].name == "starts"
                            )
                        )
                    ):




                        input_args.append(self._tuple_expr_expr(arg))
                    else:
                        input_args.append(self.expr(arg))
                else:
                    attr_idx = i - len(input_params)
                    if attr_idx >= len(attr_params):
                        raise VerifyError(
                            f"{name!r}: too many positional arguments "
                            f"(expected at most {len(input_params) + len(attr_params)}, got {len(pos_args)})"
                        )
                    attr_name = attr_params[attr_idx].name
                    if attr_name in attr_kwargs:
                        raise VerifyError(
                            f"{name!r}: duplicate binding for attribute {attr_name!r}"
                        )
                    attr_kwargs[attr_name] = self._eval_static_or_sugar(
                        attr_name, arg, schema=schema
                    )


        explicit_loc: str | None = None
        explicit_loc_given = False


        for k in node.keywords:
            if k.arg == "loc":
                explicit_loc = self._eval_static(k.value)
                explicit_loc_given = True
                continue
            if k.arg in attr_kwargs:
                raise VerifyError(
                    f"{name!r}: duplicate binding for attribute {k.arg!r} "
                    f"(both positional and keyword)"
                )
            attr_kwargs[k.arg] = self._eval_static_or_sugar(k.arg, k.value, schema=schema)



        if "storage" in attr_kwargs:
            attr_kwargs["storage"] = resolve_storage(attr_kwargs["storage"])

        op_inst = self._build_op_instance(schema, attr_kwargs)
        call = self._build_call(op_inst, tuple(input_args))
        if explicit_loc_given:
            call = replace_metadata(call, BindingMetadata(explicit_loc))
            self._explicit_binding_call_ids.add(id(call))




        self._call_dsl_names[id(call)] = schema.name
        return call

    def _build_op_instance(self, schema, attr_kwargs):
        """Construct an Op instance from a resolved schema and attr kwargs.

        There is a single path — every schema (real Op or surface
        alias) carries a ``builder`` callable. Real-Op schemas default
        to ``cls`` itself; alias schemas have a custom builder that
        constructs the kinded target Op.
        """
        return schema.builder(**attr_kwargs)



    def _maybe_autofill_binding(self, expr: Expr, name: str) -> Expr:
        """Set a Call binding label to *name* unless ``loc=`` was explicit.

        Returns *expr* unchanged when it is not a Call or already has an
        explicit binding.
        """
        if not isinstance(expr, Call):
            return expr
        if id(expr) in self._explicit_binding_call_ids:
            return expr
        return self._with_binding(expr, name)

    def _maybe_autofill_binding_default(self, expr: Expr) -> Expr:
        """Maybe autofill binding default.

        Set the Call binding label to the DSL callable name (default) when the
        user did not supply ``loc=`` explicitly. Used for tuple-unpack
        parents where there is no single LHS variable name to inherit.
        """
        if not isinstance(expr, Call):
            return expr
        if id(expr) in self._explicit_binding_call_ids:
            return expr
        dsl_name = self._call_dsl_names.get(id(expr))
        if dsl_name is None:
            return expr
        binding = get_metadata(expr, BindingMetadata)
        if binding is not None and binding.name == dsl_name:
            return expr
        return self._with_binding(expr, dsl_name)

    def _build_call(
        self, op_inst, args: tuple[Expr, ...], *, records: tuple[IRMetadata, ...] = ()
    ) -> Call:
        """Build a Call with type eagerly populated via the typeinfer registry.

        *records* are carried on the node the typeinfer walk sees, because a
        record stating how the call binds its arguments has to be there before
        the type is derived from them.
        """
        args = _with_python_float_dtypes(args)


        placeholder = Call(
            type=TensorType.scalar(DType.f32), target=op_inst, args=args,
            metadata=(*self._source_metadata(), *records),
        )
        fn = typeinfer_registry.lookup(type(op_inst))
        if fn is None:
            raise VerifyError(f"no typeinfer registered for {type(op_inst).__name__}")
        computed = fn(placeholder, self._ctx)
        return dataclasses.replace(placeholder, type=computed)

    def _eval_static_or_sugar(
        self,
        attr_name: str,
        node: ast.AST,
        *,
        schema=None,
        op_cls: type | None = None,
    ):
        """Evaluate an attribute with annotation-driven layout-sugar parsing.

        Prefer alias-aware schema annotations and retain operation classes for
        legacy callers. Without an annotation, a ``layout`` attribute still
        attempts shard-layout sugar before static evaluation. See
        [parser §4.4](docs/spec/parser.md#44-annotation-driven-sugar-dispatch).
        """
        annotation = self._lookup_param_annotation(
            schema=schema, op_cls=op_cls, attr_name=attr_name
        )
        if annotation is not None and _is_tuple_sugar(node):
            sugar = self._sugar_parser_for_annotation(annotation)
            if sugar is not None:
                try:
                    return sugar(node)
                except LayoutSugarError:


                    raise
                except ValueError:
                    pass
        elif annotation is None and attr_name == "layout" and _is_tuple_sugar(node):

            try:
                return parse_shard_layout_sugar(
                    node, self._resolve_body_mesh,
                    default_mesh=self._current_default_mesh(),
                    closure=self.closure,
                )
            except LayoutSugarError:
                raise
            except ValueError:
                pass
        value = self._eval_static(node)

        if (
            isinstance(value, str)
            and isinstance(annotation, type)
            and issubclass(annotation, enum.Enum)
        ):
            try:
                return annotation(value)
            except ValueError:
                valid = ", ".join(repr(e.value) for e in annotation)
                raise VerifyError(
                    f"{annotation.__name__}: unknown value {value!r}; "
                    f"valid values are {valid}"
                ) from None
        if isinstance(value, str) and annotation is DType:
            try:
                return DType.from_name(value)
            except ValueError as exc:
                raise VerifyError(str(exc)) from None
        return value

    def _lookup_param_annotation(
        self,
        *,
        schema=None,
        op_cls: type | None = None,
        attr_name: str,
    ) -> type | None:
        """Return the ``ParamDef.annotation`` for *attr_name*.

        Prefers the explicit ``schema`` argument (alias-aware); falls
        back to ``op_cls._op_schema.signature`` for legacy callers.
        Returns ``None`` when no schema/ParamDef matches.
        """
        if schema is None and op_cls is not None:
            schema = getattr(op_cls, "_op_schema", None)
        if schema is None:
            return None
        for pd in schema.signature:
            if pd.name == attr_name:
                return pd.annotation
        return None

    def _sugar_parser_for_annotation(self, annotation: type):
        """Return the sugar parser for a Layout-family annotation, else None."""
        if annotation is ShardLayout:
            return lambda n: parse_shard_layout_sugar(
                n, self._resolve_body_mesh,
                default_mesh=self._current_default_mesh(),
                closure=self.closure,
            )
        if annotation is Layout:
            return parse_layout_sugar
        return None

    def _resolve_static_attribute(self, owner, attr: str):
        """Resolve a static ``owner.attr`` access during ``_eval_static``.

        Default: plain ``getattr``. Dialect visitors override this to add
        context-sensitive resolution (e.g. the TIR parser checks that an MMA
        fragment ``atom.A`` is used inside a compatible enclosing mesh scope).
        """
        return getattr(owner, attr)

    def _eval_static(self, node: ast.AST, *, allow_runtime_scalar: bool = False):
        """Eval static.

        Evaluate an AST node statically for attribute kwargs (axis=1,
        new_shape=(M,K), layout=ShardLayout(...), etc.).

        Thin policy wrapper over :func:`eval_static` (parser/static_eval.py):
        the full node set, ``Name`` resolution through the lexical env before
        the closure, closure-captured-IR-object warnings, and true division
        for ``ast.Div``.
        """
        return eval_static(
            node,
            closure=self.closure,
            lookup=self.env.lookup,
            attr_resolver=self._resolve_static_attribute,
            on_closure_name=_warn_if_ir_object,
            allow_runtime_scalar=allow_runtime_scalar,
        )


__all__ = ["BaseExprVisitor", "extract_ast", "Token"]
