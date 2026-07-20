from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass
from typing import Any, Union

from tilefoundry.ir.core import Call, Expr, Var, VerifyError
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.tir.launch import LaunchAttrs, launch_call
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.shape import shape_var_name
from tilefoundry.ir.tir.stmt import Stmt
from tilefoundry.ir.tir.stmts import (
    Evaluate,
    For,
    If,
    LetStmt,
    MeshScope,
    Return,
    Sequential,
    While,
)
from tilefoundry.ir.tir.symbol_ref import symbol_call
from tilefoundry.ir.types import DType, TensorType, UnitType
from tilefoundry.ir.types.dim import (
    DimVar,
    is_dim_expr,
)
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.visitor import StmtVisitor
from tilefoundry.target import CudaTarget, default_target

from .base import BaseExprVisitor, _i64, extract_ast
from .dispatch import resolve_stmt
from .symtab import LexicalEnv


@dataclass(frozen=True)
class _Bind:
    var: Var
    value: Expr


_Item = Union[_Bind, Stmt]


def _fold_items(items: list[_Item]) -> Sequential:
    """Fold ``_Bind`` markers into nested ``LetStmt``; plain Stmts stay."""
    def fold(i: int) -> list[Stmt]:
        out: list[Stmt] = []
        while i < len(items):
            item = items[i]
            if isinstance(item, _Bind):
                inner = fold(i + 1)
                out.append(
                    LetStmt(
                        var=item.var,
                        value=item.value,
                        body=Sequential(body=tuple(inner)),
                    )
                )
                return out
            out.append(item)
            i += 1
        return out

    return Sequential(body=tuple(fold(0)))


def _is_device_target(target) -> bool:
    """A device (kernel) target — anything but the host ``cpu`` target. Host
    entries read shapes from their tensor args at runtime, so they never carry
    hidden shape scalars."""
    return target is not None and getattr(target, "name", None) != "cpu"


def _dim_var_names_in_type(ty: Any) -> set[str]:
    """DimVar names appearing in a ``TensorType``'s shape (and the shape of its
    ``ShardLayout``'s inner layout)."""
    names: set[str] = set()
    for d in getattr(ty, "shape", None) or ():
        if isinstance(d, DimVar):
            names.add(d.name)
    inner = getattr(getattr(ty, "layout", None), "layout", None)
    for d in getattr(inner, "shape", None) or ():
        if isinstance(d, DimVar):
            names.add(d.name)
    return names


class _DimVarRefCollector(StmtVisitor):
    """Collect DimVar names the body actually references via the types of bound
    results and op operands. A dynamic tensor dim surfaces in these
    ``TensorType`` shapes; the codegen plumbs each as a ``<param>_shape_<axis>``
    runtime scalar, so the kernel signature must declare it."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_LetStmt(self, s: LetStmt) -> None:
        self.names |= _dim_var_names_in_type(getattr(s.var, "type", None))
        self.generic_visit(s)

    def visit_Evaluate(self, s: Evaluate) -> None:
        for a in s.args:
            self.names |= _dim_var_names_in_type(getattr(a, "type", None))
        self.generic_visit(s)


def _shape_scalar_params(
    params: tuple[Var, ...], referenced: set[str]
) -> tuple[Var, ...]:
    """Hidden ``<param>_shape_<axis>`` i32 scalar params for each referenced
    dynamic dim, mirroring the HIR→TIR lowering. Each DimVar maps to its first
    occurrence in a tensor param's shape (the same rule codegen uses to source
    the runtime extent). Idempotent: a param that already exists is skipped."""
    existing = {p.name for p in params}
    scalar_i32 = TensorType.scalar(dtype=DType.i32)
    seen: set[str] = set()
    extra: list[Var] = []
    for p in params:
        ty = p.type
        if not isinstance(ty, TensorType):
            continue
        for axis, dim in enumerate(ty.shape):
            if not isinstance(dim, DimVar) or dim.name not in referenced:
                continue
            if dim.name in seen:
                continue
            seen.add(dim.name)
            name = shape_var_name(p.name, axis)
            if name not in existing:
                extra.append(Var(type=scalar_i32, name=name))
    return tuple(extra)


def parse_prim_func(fn, *, target=None, extra_closure=None) -> PrimFunction:
    node = extract_ast(fn)
    closure = _collect_closure(fn, extra_closure)
    env = LexicalEnv()
    params = _build_params(node, closure)
    for p in params:
        env.define(p.name, p)
    visitor = _TirBodyVisitor(env, closure)
    body = _fold_items(visitor.visit_body(node.body))
    # A device kernel that reads a dynamic tensor dim (a DimVar axis) needs the
    # runtime extent plumbed as a hidden ``<param>_shape_<axis>`` i32 scalar —
    # the same ABI the HIR→TIR lowering appends. Host entries read shapes from
    # their tensor args, so they get no hidden scalars.
    if _is_device_target(target):
        collector = _DimVarRefCollector()
        collector.visit(body)
        params = (*params, *_shape_scalar_params(params, collector.names))
    kwargs = {} if target is None else {"target": target}
    return PrimFunction(name=node.name, params=params, body=body, **kwargs)


def _collect_closure(fn, extra_closure=None) -> dict[str, Any]:
    closure: dict[str, Any] = {}
    # ``extra_closure`` (sibling IR functions from a ``@module`` class body)
    # sits below the function's own globals/freevars so it cannot shadow them.
    if extra_closure:
        closure.update(extra_closure)
    if fn.__globals__ is not None:
        closure.update(fn.__globals__)
    if fn.__closure__ is not None:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            try:
                closure[name] = cell.cell_contents
            except ValueError:
                pass
    return closure


def _build_params(node: ast.FunctionDef, closure: dict[str, Any]) -> tuple[Var, ...]:
    out: list[Var] = []
    for a in node.args.args:
        if a.annotation is None:
            raise VerifyError(f"@tilefoundry.prim_func param {a.arg!r} must be annotated")
        code = compile(ast.Expression(body=a.annotation), "<ann>", "eval")
        val = eval(code, closure)  # noqa: S307
        if not isinstance(val, TensorType):
            raise VerifyError(
                f"param {a.arg!r}: annotation did not resolve to TensorType, got {type(val).__name__}"
            )
        out.append(Var(type=val, name=a.arg))
    return tuple(out)


def _is_none(node: ast.AST) -> bool:
    """True for a literal ``None`` AST node."""
    return isinstance(node, ast.Constant) and node.value is None


# Accepted keyword arguments of the ``launch(...)`` surface.
_LAUNCH_CONFIG_KEYS = frozenset(
    {"grid", "block", "cluster", "dynamic_smem", "stream", "attrs"}
)


class _TirBodyVisitor(BaseExprVisitor):
    token = "tir"

    def visit_body(self, stmts: list[ast.stmt]) -> list[_Item]:
        out: list[_Item] = []
        for node in stmts:
            s = self._visit_stmt(node)
            if s is not None:
                out.append(s)
        return out

    def _visit_stmt(self, node: ast.stmt) -> _Item | None:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise VerifyError("tir: only single-target Name assignments supported in V1")
            tgt = node.targets[0].id
            # Compile-time static binding: a platform-namespace descriptor
            # (`op = T.cuda.mma.<NAME>` / `atom = T.cuda.mma.atom(op=...)`).
            # Bound in the parser env as a Python object, NOT a runtime
            # LetStmt — subsequent `atom.A/B/C` resolve statically.
            if self._is_platform_rooted(node.value):
                self.env.define(tgt, self._eval_static(node.value))
                return None
            rhs = self.expr(node.value)
            var = Var(type=rhs.type, name=tgt)
            self.env.define(tgt, var)
            # Parser surface `x = rhs` lowers to LetStmt via fold; yield a
            # bind marker.
            return _Bind(var=var, value=rhs)

        if isinstance(node, ast.Expr):
            inner = node.value
            if not isinstance(inner, ast.Call):
                raise VerifyError("tir: bare expression statement must be a call")
            return self._call_as_top_level_stmt(inner)

        if isinstance(node, ast.Return):
            if node.value is not None:
                raise VerifyError("§8.3: @tilefoundry.prim_func return must be bare (no value)")
            return Return()

        if isinstance(node, ast.For):
            return self._visit_for(node)

        if isinstance(node, ast.While):
            cond = self.expr(node.test)
            body = _fold_items(self.visit_body(node.body))
            return While(cond=cond, body=body)

        if isinstance(node, ast.If):
            cond = self.expr(node.test)
            then_body = _fold_items(self.visit_body(node.body))
            else_body = _fold_items(self.visit_body(node.orelse)) if node.orelse else Sequential(body=())
            return If(cond=cond, then_body=then_body, else_body=else_body)

        if isinstance(node, ast.With):
            return self._visit_with(node)

        raise VerifyError(f"tir: unsupported statement {type(node).__name__}")

    def _call_as_top_level_stmt(self, node: ast.Call) -> Stmt:
        # Name-only special forms: launch, Stmt classes / intrinsics, and
        # sibling ``@prim_func`` callees.
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "launch":
                return self._launch_as_stmt(node)
            stmt_cls = resolve_stmt(name)
            if stmt_cls is not None:
                kwargs = {k.arg: self._eval_static(k.value) for k in node.keywords}
                pos = [self.expr(a) for a in node.args]
                field_names = [f.name for f in dataclasses.fields(stmt_cls) if f.name != "loc"]
                if len(pos) > len(field_names):
                    raise VerifyError(f"tir: {name!r} too many positional args")
                bound = dict(zip(field_names, pos))
                bound.update(kwargs)
                return stmt_cls(**bound)
            # ``@prim_func`` evaluates to the PrimFunction directly, so a sibling
            # callee binding *is* that IR function.
            callee_ir = self.closure.get(name)
            if isinstance(callee_ir, PrimFunction):
                if node.keywords:
                    raise VerifyError(f"tir: call to {name!r} does not support kwargs")
                args = tuple(self.expr(a) for a in node.args)
                return symbol_call(callee_ir, args)
        # Effect-form op statement — bare ``copy(...)`` or namespaced
        # ``T.copy(...)`` / ``T.mma(...)``. The op call is unit-typed; wrap it in
        # ``Evaluate``. A value op (non-unit) at Stmt position is an error.
        expr = self.call_to_op_call(node)
        if isinstance(expr, Call) and isinstance(expr.type, UnitType):
            return Evaluate(callable=expr.target, args=expr.args)
        disp = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
        raise VerifyError(
            f"tir: §8.5 value op {disp!r} cannot be top-level Stmt; wrap with `=`"
        )

    def _is_platform_rooted(self, node: ast.AST) -> bool:
        """True when ``node`` is a ``T.<platform>...`` expression — an
        attribute/call chain whose root Name resolves to the ``dsl.T`` module
        and whose first attribute is a platform name (``cuda``, later other
        targets). Such expressions are compile-time descriptors
        (``MmaOpSpec`` / ``MmaAtom``), bound statically rather than lowered to
        a ``LetStmt``."""
        cur = node
        first_attr_on_root: str | None = None
        while isinstance(cur, (ast.Attribute, ast.Call)):
            if isinstance(cur, ast.Call):
                cur = cur.func
            else:
                first_attr_on_root = cur.attr
                cur = cur.value
        if not isinstance(cur, ast.Name):
            return False
        root = self.env.lookup(cur.id)
        if root is None:
            root = self.closure.get(cur.id)
        import tilefoundry.dsl as _dsl  # noqa: PLC0415
        if root is not _dsl.T:
            return False
        from tilefoundry.dsl.T._platforms import PLATFORM_NAMESPACES  # noqa: PLC0415
        return first_attr_on_root in PLATFORM_NAMESPACES

    def _launch_as_stmt(self, node: ast.Call) -> Evaluate:
        # launch(device_fn, *tensor_args, grid=, block=, cluster=None,
        #        dynamic_smem=0, stream=None, attrs=...)
        if not node.args:
            raise VerifyError("tir: launch(...) needs a device function first argument")
        callee_node = node.args[0]
        if not isinstance(callee_node, ast.Name):
            raise VerifyError("tir: launch(...) first argument must be a function name")
        callee_ir = self.closure.get(callee_node.id)
        # The callee may be an already-lowered cuda PrimFunction or an HIR
        # @func device function (lowered later — HirToTir rewrites the callee).
        if not isinstance(callee_ir, (HirFunction, PrimFunction)):
            raise VerifyError(
                f"tir: launch(...) callee {callee_node.id!r} must be a @func or "
                f"@prim_func device function"
            )
        effective_target = (
            callee_ir.target
            if callee_ir.target is not None
            else default_target()
        )
        if not isinstance(effective_target, CudaTarget):
            raise VerifyError(
                f"tir: launch(...) callee {callee_node.id!r} must target a CUDA device"
            )
        # Tensor args only; launch config lives in the keyword arguments.
        args = tuple(self.expr(a) for a in node.args[1:])
        kw: dict[str, ast.AST] = {}
        for k in node.keywords:
            if k.arg is None:
                raise VerifyError("tir: launch(...) does not accept `**kwargs`")
            if k.arg not in _LAUNCH_CONFIG_KEYS:
                raise VerifyError(
                    f"tir: launch(...) got unexpected keyword {k.arg!r}; "
                    f"allowed: {sorted(_LAUNCH_CONFIG_KEYS)}"
                )
            kw[k.arg] = k.value
        if "grid" not in kw or "block" not in kw:
            raise VerifyError("tir: launch(...) requires `grid=` and `block=`")
        cluster = (
            self._launch_dim(kw["cluster"])
            if "cluster" in kw and not _is_none(kw["cluster"])
            else None
        )
        dynamic_smem = self.expr(kw["dynamic_smem"]) if "dynamic_smem" in kw else 0
        stream = (
            self.expr(kw["stream"])
            if "stream" in kw and not _is_none(kw["stream"])
            else None
        )
        if "attrs" in kw:
            attrs = self._eval_static(kw["attrs"])
            if not isinstance(attrs, LaunchAttrs):
                raise VerifyError(
                    f"tir: launch(...) `attrs=` must be a LaunchAttrs, got "
                    f"{type(attrs).__name__}"
                )
        else:
            attrs = LaunchAttrs()
        return launch_call(
            callee_ir,
            args,
            self._launch_dim(kw["grid"]),
            self._launch_dim(kw["block"]),
            cluster=cluster,
            dynamic_smem=dynamic_smem,
            stream=stream,
            attrs=attrs,
        )

    def _launch_dim(self, value: ast.AST) -> tuple[Expr, Expr, Expr]:
        """Normalize a launch grid / block spec to a 3-tuple of extents.

        Grid / block extents are compile-time shape arithmetic, not runtime
        expressions, so each element is evaluated statically (a literal, a
        ``DimVar``, or a dim-arithmetic ``Expr`` such as ``ceildiv(S, tile)``).
        A scalar or 1-/2-tuple is right-padded with the constant ``1``.
        """
        nodes = value.elts if isinstance(value, ast.Tuple) else [value]
        elts = [self._eval_launch_extent(n) for n in nodes]
        if len(elts) > 3:
            raise VerifyError("tir: launch grid/block accepts at most 3 dimensions")
        while len(elts) < 3:
            elts.append(_i64(1))
        return tuple(elts)

    def _eval_launch_extent(self, node: ast.AST):
        """Statically evaluate one grid / block extent to an ``int`` (wrapped as
        a constant), a ``DimVar``, or a dim-arithmetic ``Expr``; reject anything
        else loudly (extents are shape arithmetic, not arbitrary values)."""
        val = self._eval_static(node)
        if not is_dim_expr(val):
            raise VerifyError(
                f"tir: launch grid/block extent must be an int, DimVar, or dim "
                f"expression, got {type(val).__name__}"
            )
        return _i64(val) if isinstance(val, int) else val

    def _visit_for(self, node: ast.For) -> For:
        if not isinstance(node.target, ast.Name):
            raise VerifyError("tir: For target must be a Name")
        if not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name):
            raise VerifyError("tir: For iter must be a plain `range(...)` call")
        if node.iter.func.id != "range":
            raise VerifyError(f"tir: For iter must be `range(...)`, got {node.iter.func.id!r}")
        args = node.iter.args
        if len(args) == 1:
            start, stop, step = _i64(0), self.expr(args[0]), _i64(1)
        elif len(args) == 2:
            start, stop, step = self.expr(args[0]), self.expr(args[1]), _i64(1)
        elif len(args) == 3:
            start, stop, step = self.expr(args[0]), self.expr(args[1]), self.expr(args[2])
        else:
            raise VerifyError("tir: range() expects 1-3 args")
        iv = Var(type=TensorType.scalar(DType.i64), name=node.target.id)
        self.env.push_frame()
        try:
            self.env.define(node.target.id, iv)
            body = _fold_items(self.visit_body(node.body))
        finally:
            self.env.pop_frame()
        return For(induction_var=iv, start=start, stop=stop, step=step, body=body)

    def _visit_with(self, node: ast.With) -> MeshScope:
        if len(node.items) != 1:
            raise VerifyError("tir: only single-item `with` supported")
        item = node.items[0]
        mesh = self._eval_static(item.context_expr)
        if not isinstance(mesh, Mesh):
            raise VerifyError(
                f"tir: `with` context must evaluate to a Mesh, got {type(mesh).__name__}"
            )
        if item.optional_vars is None or not isinstance(item.optional_vars, ast.Name):
            raise VerifyError("tir: `with Mesh(...) as name` requires a single Name binding")
        binding_name = item.optional_vars.id
        binding = Var(type=TensorType.scalar(DType.i64), name=binding_name)
        self.env.push_frame()
        try:
            self.env.define(binding_name, mesh)
            body = _fold_items(self.visit_body(node.body))
        finally:
            self.env.pop_frame()
        return MeshScope(mesh=mesh, binding=binding, body=body)

    def _resolve_static_attribute(self, owner, attr: str):
        """TIR static attribute resolution.

        An MMA fragment access ``atom.A/B/C`` returns the atom's layout
        contract **as-is** (no rebind). But because a fragment is only valid in
        a thread scope that can host the atom, we check the enclosing mesh
        scope against ``atom.required_scope`` here, at the use point — rejecting
        e.g. a ``cta`` or wrong-sized ``thread`` scope. The match is structural
        (thread participation), independent of binding/axis names.
        """
        from tilefoundry.ir.tir.cuda.nn.mma_atom import MmaAtom  # noqa: PLC0415
        from tilefoundry.ir.types.shard.scope_match import (  # noqa: PLC0415
            mesh_scope_matches_required_scope,
        )

        val = getattr(owner, attr)
        if isinstance(owner, MmaAtom) and attr in ("A", "B", "C"):
            mesh = self._current_default_mesh()
            if mesh is None:
                raise VerifyError(
                    f"mma fragment `atom.{attr}` must be used inside a "
                    f"`with Mesh(...)` thread scope"
                )
            if not mesh_scope_matches_required_scope(mesh, owner.required_scope):
                req = owner.required_scope
                raise VerifyError(
                    f"mma fragment `atom.{attr}`: enclosing mesh scope does not "
                    f"match the atom's required thread scope "
                    f"(topology {req.topology.name!r}, {req.topology.size} lanes); "
                    f"a {req.topology.name}({req.topology.size}) scope with the "
                    f"exact lane layout shape {tuple(req.layout.shape)} strides "
                    f"{tuple(req.layout.strides)} is required (e.g. the fragment "
                    f"`Split` axes need the 2-axis (4,8) warp, not a flat (32,))"
                )
        return val

    def visit_Call(self, node: ast.Call) -> Expr:
        return self.call_to_op_call(node)


__all__ = ["parse_prim_func"]
