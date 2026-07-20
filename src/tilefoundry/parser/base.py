from __future__ import annotations

import ast
import dataclasses
import enum
import inspect
import logging
import textwrap
from typing import Any, Callable

from tilefoundry.ir.core import Call, Constant, Expr, Tuple, TypeInferContext, Var, VerifyError
from tilefoundry.ir.core.op_schema import OpSchema
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.function import elaborate
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.ir.types.shape_helpers import i64_const
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.shard_layout import ShardLayout
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
from .range_slice import RangeSlice
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

# IR object types that should be in DSL source, not closure
_IR_OBJECT_TYPES = {
    "Topology": None,
    "Mesh": None,
    "MeshAxis": None,
    "ShardLayout": None,
    "Layout": None,
}


def _warn_if_ir_object(val: Any, name: str) -> None:
    """Warn when a preconstructed IR object is resolved from closure.

    Canonical DSL source should use AST constructor syntax
    (e.g. ``Topology("cta", 128)``) or topology-name string resolution
    (e.g. ``with Mesh(topology="cta", ...)``) instead of capturing
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
    src = textwrap.dedent(inspect.getsource(fn))
    mod = ast.parse(src)
    # decorator may wrap but we parse the source including decorator; pick
    # the first FunctionDef found.
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
    """Return the subscript base identifier (``Tensor`` / ``ConstTensor``),
    resolving through an attribute path such as ``dsl.ConstTensor``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_const_tensor_annotation(node: ast.AST) -> bool:
    """``ConstTensor[...]`` marks a parameter ``is_const=True``; ``Tensor[...]``
    and every other annotation form leave it ``False``."""
    return (
        isinstance(node, ast.Subscript)
        and _annotation_head_name(node.value) == "ConstTensor"
    )


def _resolve_tensor_type(node: ast.AST, closure: dict[str, Any]) -> TensorType:
    """Resolve a tensor type annotation. Shared by ``@func`` and
    ``@prim_func`` param / return annotations (parser.md §1.4) so a layout-
    sugar ``Tensor[...]`` annotation resolves identically in both dialects.

    Supports two forms:

    1. **Sugar**: ``Tensor[(M,K), bf16, ((32 @ gpu.cluster, K), {gpu.warp @ P("sum")}), "smem"]``
       — compact layout sugar, parsed directly from the AST without ``eval()``.
    2. **Verbose**: ``Tensor[(M,K), bf16, ShardLayout(...), "smem"]``
       — evaluated via ``eval()`` in *closure*.
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
    # A source value literal is unmaterialized (storage=umat): it carries no
    # committed memory residency until a use site or lowering fixes it.
    if isinstance(value, bool):
        return Constant(type=TensorType.scalar(DType.bool, storage=StorageKind.UMAT), value=value)
    if isinstance(value, int):
        return Constant(type=TensorType.scalar(DType.i64, storage=StorageKind.UMAT), value=value)
    if isinstance(value, float):
        return Constant(type=TensorType.scalar(DType.f32, storage=StorageKind.UMAT), value=value)
    raise VerifyError(f"unsupported literal type {type(value).__name__}")


class BaseExprVisitor:
    """Shared visitor for Expr-returning AST nodes. Emits core_ir Expr."""

    token: Token

    def __init__(self, env: LexicalEnv, closure: dict[str, Any]):
        self.env = env
        self.closure = closure  # function's captured globals + nonlocals
        # Shared TypeInferContext so each Call's .type is filled eagerly at
        # parse time (callers need accurate types for subsequent Assign Var
        # construction / tir verify / etc.).
        self._ctx = TypeInferContext()
        # Track which Call nodes were assigned a loc explicitly via
        # a user-supplied ``loc=...`` kwarg. The Assign handler suppresses
        # LHS-name auto-fill for those Calls (explicit value wins).
        self._explicit_loc_call_ids: set[int] = set()
        # DSL callable name used to instantiate each Call. Provides a
        # readable default loc for tuple-unpack parents (`rope` → "rope") when
        # the user did not supply ``loc=...`` and there is no single LHS name
        # to fall back on.
        self._call_dsl_names: dict[int, str] = {}

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

    # ---- Expr-returning dispatch ----------------------------------------------------

    def expr(self, node: ast.AST) -> Expr:
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise VerifyError(f"unsupported AST node in expression: {type(node).__name__}")
        return method(node)

    # Constants ----------------------------------------------------------------------

    def visit_Constant(self, node: ast.Constant) -> Expr:
        return _constant_from_py(node.value)

    # Names --------------------------------------------------------------------------

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
        if isinstance(val, (int, float, bool)):
            return _constant_from_py(val)
        # Check for IR objects resolved from closure (not lexical env):
        # these are not canonical — they should be in DSL source
        if from_closure and type(val).__name__ in _IR_OBJECT_TYPES:
            _warn_if_ir_object(val, node.id)
        raise VerifyError(f"name {node.id!r} resolved to non-Expr Python value {type(val).__name__}")

    # Attribute access (cta.x etc.) --------------------------------------------------

    def visit_Attribute(self, node: ast.Attribute) -> Expr:
        # `cta.x` / `cta.y` resolves through the lexical env to a
        # compile-time MeshAxis (Python object). It is NOT an Expr — callers
        # embedding it (e.g. ShardLayout construction) handle that. If it
        # really reaches this Expr dispatcher, raise.
        raise VerifyError(f"attribute access {ast.unparse(node)!r} not valid as Expr")

    # Subscript: only TupleType, integer-constant index — emits TupleGetItem -------

    def visit_Subscript(self, node: ast.Subscript) -> Expr:
        """Resolve ``expr[idx]`` to a ``TupleGetItem`` or ``Slice`` Call.

        - ``TupleType`` value + int constant index → ``TupleGetItem``.
        - ``TensorType`` value + slice/RangeSlice indices → ``Slice``.
          Each dim accepts either an ``ast.Slice``
          (full / start:stop[:step]) or a Name resolving to a
          ``RangeSlice`` (from ``for ok in tile(extent, step)``).
          Plain integer indexing collapses dims and is not yet
          supported here.
        """
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
        - an ``ast.Name`` resolving to a ``RangeSlice`` parser-side
          binding (``for ok in tile(extent, step)``).

        Other forms (constants, computed Expr indices, ellipsis, lists)
        are deferred to gather/scatter ops and raise here.
        """
        # Normalize to a list of dim slicers.
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

        begin: list[Any] = []
        end: list[Any] = []
        strides: list[Any] = []
        for axis, (el, dim) in enumerate(zip(elts, x_ty.shape)):
            b, e, s = self._slicer_for_dim(el, dim, axis)
            begin.append(b)
            end.append(e)
            strides.append(s)

        return self._build_call(
            Slice(begin=tuple(begin), end=tuple(end), strides=tuple(strides)),
            (value,),
        )

    def _slicer_for_dim(self, el: ast.AST, dim: Any, axis: int):
        """Resolve one subscript element to ``(begin, end, stride)``.

        ``dim`` is the input tensor's static shape value at this axis
        (used as the default upper bound for ``:``).
        """
        if isinstance(el, ast.Slice):
            # full slice ``:`` or partial ``a:b[:c]``
            if el.lower is None:
                begin = 0
            else:
                begin = self._eval_static(el.lower)
            if el.upper is None:
                end = dim
            else:
                end = self._eval_static(el.upper)
            if el.step is None:
                stride = 1
            else:
                stride = self._eval_static(el.step)
            return begin, end, stride
        if isinstance(el, ast.Name):
            val = self.env.lookup(el.id)
            if isinstance(val, RangeSlice):
                return val.start, val.stop, 1
        raise VerifyError(
            f"tensor subscript axis {axis}: unsupported indexer "
            f"{ast.dump(el)} (expected `:`, `a:b`, or a tile RangeSlice)"
        )

    # Binary ops ---------------------------------------------------------------------

    def visit_BinOp(self, node: ast.BinOp) -> Expr:
        opname = type(node.op).__name__
        # MatMult (``@``) routes to the dedicated MatMul Op (not kinded).
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

    # Compare: Python allows `a < b < c` but V1 only supports pairwise --------------

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

    # UnaryOp (Neg / Not) -------------------------------------------------------------

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Expr:
        opname = type(node.op).__name__
        kind = _unary_kind_for_ast_op(opname)
        if kind is None:
            raise VerifyError(f"unsupported unary op {opname}")
        operand = self.expr(node.operand)
        return self._build_call(self._make_unary(kind), (operand,))

    def visit_Call(self, node: ast.Call) -> Expr:
        return self.call_to_op_call(node)

    # Generic call -------------------------------------------------------------------

    def _resolve_call_target(self, func: ast.AST):
        """Resolve the callee AST node to an ``OpSchema``, or ``None``.

        This returns ``OpSchema`` instances rather than bare ``Op``
        classes so surface aliases (``schema.op_class is None``) flow
        through the same parser path as real Ops via ``schema.builder``.

        Two callee forms are accepted:

        - ``ast.Name``: bare ``add(...)`` — looked up against the
          parser's lexical environment + the function's closure first. The
          bound value must carry an ``_op_schema`` attribute (set by
          ``@register_op`` on Op classes and by ``@register_alias``
          on alias builder functions, both of which
          ``tilefoundry.dsl.tf.<name>`` returns). When the closure path
          misses, ``dispatch.resolve_callable`` is consulted (parser.md
          §3.2/§3.3) — dialect-strict registry dispatch honouring the
          TIR-only trailing-underscore effect-form selector (§1.3/§4.6).
        - ``ast.Attribute(value=ast.Name(<ns>))``: ``tf.add(...)``
          / ``T.copy(...)``. The leading Name resolves to the
          ``tilefoundry.dsl.tf`` / ``T`` namespace module (matched by
          identity); the attribute name is then dispatched against
          the matching dialect's OpSchema registry via
          ``dispatch.resolve_schema``, alias-aware.
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
            # Match by module identity to avoid catching arbitrary
            # objects that happen to expose the op name.
            # noqa cycle: tilefoundry.dsl pulls tilefoundry.parser.overload, which
            # would re-enter this module at import time.
            import tilefoundry.dsl as _dsl  # noqa: PLC0415
            if ns is _dsl.tf:
                return resolve_schema(func.attr, "tf")
            if ns is _dsl.T:
                return resolve_schema(func.attr, "T")
        return None

    def _resolve_function_target(self, func: ast.AST):
        """Return the ``hir.Function`` instance behind a callee AST, or
        ``None`` when the callee is not an ``@func``-decorated function.
        ``@func`` evaluates to the ``hir.Function`` directly, so a sibling
        callee binding *is* that Function (see :func:`tilefoundry.script.func`).
        """
        val: Any = None
        if isinstance(func, ast.Name):
            val = self.env.lookup(func.id)
            if val is None:
                val = self.closure.get(func.id)
        # Attribute-style HIR Function calls (``mod.fn(...)``) are out
        # of scope for v0; require a bare name binding so the closure
        # lookup is unambiguous.
        if isinstance(val, HirFunction):
            return val
        return None

    def _build_function_call(
        self, callee: Any, node: ast.Call, name: str
    ) -> Expr:
        """Build a ``Call(target=<hir.Function>, args=...)`` for a nested
        ``@func`` → ``@func`` call site. Arg-count enforcement lives in
        the parser; argument *types* are bound by elaboration
        (``tilefoundry.ir.hir.function.elaborate``, hir.md §1.1) before the
        ``Call`` is built, so ``Call.target`` is the actual per-call-site
        instance (needed for the viewer/printer to read correctly-propagated
        types off ``call.target.body``), not just ``Call.type``.
        ``loc=`` keyword is accepted and threaded onto ``Call.loc``;
        every other keyword is rejected because hir Function calls are
        positional-only at the IR level.
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
        expected = len(callee.params)
        got = len(node.args)
        if got != expected:
            raise VerifyError(
                f"{name!r}: nested @func call arity mismatch — callee "
                f"declares {expected} parameter(s), call passed {got}"
            )
        input_args = tuple(self.expr(a) for a in node.args)
        # The real Call is built from `instance` below, so it doesn't exist
        # yet at this point; a surrogate carrying the same loc an explicit
        # `loc=` keyword would produce lets an arity/bind error report
        # `at <loc>` instead of the callee's (always-None) own `.loc`.
        call_for_errors = Call(
            type=callee.return_type, target=callee, args=input_args,
            loc=explicit_loc if explicit_loc_given else None,
        )
        instance = elaborate(
            callee, tuple(a.type for a in input_args), self._ctx,
            call=call_for_errors,
        )
        call = self._build_call(instance, input_args)
        if explicit_loc_given:
            call = dataclasses.replace(call, loc=explicit_loc)
            self._explicit_loc_call_ids.add(id(call))
        # Default loc fallback uses the surface name.
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
        # Surface display name for error messages and effect-stmt detection.
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = ast.unparse(node.func)
        else:
            raise VerifyError("only Name / Attribute callees supported in V1")

        # ---- nested @func → @func call --------------------------------
        # Look the callee up before schema dispatch. A name bound to an
        # ``hir.Function`` (``@func`` evaluates to one) is the target of the
        # resulting ``Call``.
        callee_func = self._resolve_function_target(node.func)
        if callee_func is not None:
            return self._build_function_call(callee_func, node, name)

        schema = self._resolve_call_target(node.func)
        if schema is None:
            # Effect stmt disguised as value call → error at Expr position.
            if isinstance(node.func, ast.Name) and resolve_stmt(name) is not None:
                raise VerifyError(
                    f"{name!r} is an effect Stmt op; cannot appear in Expr position "
                    f"(wrap in Assign or use as top-level Stmt)"
                )
            raise VerifyError(f"unknown Op name {name!r}")
        # Build parameter info for positional attr binding from schema signature
        # (works uniformly for real-Op schemas and surface aliases).
        param_infos = schema.signature
        input_params = [p for p in param_infos if p.kind == "input"]
        attr_params = [p for p in param_infos if p.kind == "attribute"]

        # Variadic-input ops (``Concat`` / ``Stack`` / ``ShapeCompose``):
        # the schema declares a single ``input`` ParamDef but the op
        # accepts any number of tensor operands. All positional args
        # bind to that single input list; attributes must be passed
        # as keyword. Detected by ``is_variadic`` on the Op class
        # carried by the schema (real-Op schemas) or by the alias
        # builder's wrapped Op class.
        is_variadic = bool(getattr(getattr(schema, "op_class", None), "is_variadic", False))

        # Positional args: first N bind to input params, remaining to attr params
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
                        and schema.name == "insert_slice"
                        and input_params[i].name == "offsets"
                    ):
                        # Narrow route: only ``insert_slice``'s per-axis offset
                        # tuple is lifted to an explicit core Tuple of
                        # scalar Exprs. Any other input keeps the default path,
                        # so a tuple literal there is rejected.
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

        # Extract user-supplied ``loc=`` kwarg (not an Op attr; lives
        # on Call). Skipped from attr binding; passed through to _build_call.
        explicit_loc: str | None = None
        explicit_loc_given = False

        # Keyword args: check for duplicates with positional attrs
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

        # Normalise a ``storage`` attribute to StorageKind | None at this
        # surface boundary so legacy string aliases never enter the IR.
        if "storage" in attr_kwargs:
            attr_kwargs["storage"] = resolve_storage(attr_kwargs["storage"])

        op_inst = self._build_op_instance(schema, attr_kwargs)
        call = self._build_call(op_inst, tuple(input_args))
        if explicit_loc_given:
            call = dataclasses.replace(call, loc=explicit_loc)
            self._explicit_loc_call_ids.add(id(call))
        # Stash DSL callable name as a default-loc fallback for downstream
        # auto-fill (e.g. tuple-unpack parent default).  Must be after any
        # ``dataclasses.replace`` so we record the final Call's id. Use
        # the schema's canonical name so loc tags stay terse regardless
        # of Name vs Attribute callee form.
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

    # Call.loc auto-fill helpers ---------------------------------

    def _maybe_autofill_loc(self, expr: Expr, name: str) -> Expr:
        """Set ``Call.loc`` to *name* when *expr* is a Call without an
        explicit loc (explicit user-supplied ``loc=`` is preserved).

        Returns *expr* unchanged when it is not a Call or already has an
        explicit loc.
        """
        if not isinstance(expr, Call):
            return expr
        if id(expr) in self._explicit_loc_call_ids:
            return expr
        if expr.loc is not None and expr.loc != self._call_dsl_names.get(id(expr)):
            # Already auto-filled with a non-default tag; keep it.
            return expr
        return dataclasses.replace(expr, loc=name)

    def _maybe_autofill_loc_default(self, expr: Expr) -> Expr:
        """Set ``Call.loc`` to the DSL callable name (default) when the
        user did not supply ``loc=`` explicitly. Used for tuple-unpack
        parents where there is no single LHS variable name to inherit.
        """
        if not isinstance(expr, Call):
            return expr
        if id(expr) in self._explicit_loc_call_ids:
            return expr
        dsl_name = self._call_dsl_names.get(id(expr))
        if dsl_name is None:
            return expr
        if expr.loc == dsl_name:
            return expr
        return dataclasses.replace(expr, loc=dsl_name)

    def _build_call(self, op_inst, args: tuple[Expr, ...]) -> Call:
        """Build a Call with type eagerly populated via the typeinfer registry."""
        # Construct with a placeholder; the registry reads (call.target, args)
        # and doesn't need call.type, so we can fix it post-hoc via dataclasses.replace.
        placeholder = Call(type=TensorType.scalar(DType.f32), target=op_inst, args=args)
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
        """Evaluate a static attribute value, with layout sugar detection.

        Sugar dispatch is annotation-driven: when the resolved schema
        has a ParamDef whose ``annotation`` is in the Layout family
        (``ShardLayout`` / ``Layout``), tuple-sugar
        literals are parsed via the corresponding sugar parser. Falls
        back to ``_eval_static()`` when no annotation hint applies or
        sugar parsing fails.

        ``schema`` is the alias-aware OpSchema (preferred). ``op_cls``
        remains accepted for legacy callers.

        Legacy heuristic: if no annotation hint is found and
        *attr_name* is literally ``"layout"`` with a tuple-sugar node,
        the ShardLayout sugar parser is still tried — this preserves
        compatibility with ops not yet migrated to ParamDef.
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
                    # Node is recognized layout sugar but malformed — surface the
                    # real diagnostic instead of masking it with _eval_static.
                    raise
                except ValueError:
                    pass
        elif annotation is None and attr_name == "layout" and _is_tuple_sugar(node):
            # Legacy fallback: pre-ParamDef ops use attr name ``layout``.
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
        # String-enum sugar: promote plain strings to the receiving enum member.
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

    def _eval_static(self, node: ast.AST):
        """Evaluate an AST node statically for attribute kwargs (axis=1,
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
        )


__all__ = ["BaseExprVisitor", "extract_ast", "Token"]
