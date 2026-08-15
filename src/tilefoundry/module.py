"""``@module`` decorator — collect a class body into an IR ``Module``."""

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass, field
from types import FrameType


class _Undeclared:
    """Distinguish an omitted topology declaration from an explicit empty one.

    An explicit empty ``topologies`` declares a topology-free execution domain,
    while omitting it declares nothing and inherits from the owning Module. A
    plain ``None`` default cannot express both, because ``None`` is already the
    IR's encoding of "inherit". This applies to both declaration surfaces: the
    ``topologies=`` arguments of ``@module`` and standalone ``@func``.
    """

    def __repr__(self) -> str:
        return "UNDECLARED"


UNDECLARED = _Undeclared()


@dataclass(eq=False)
class _Entry:
    topologies: tuple | None
    frame: FrameType
    owner_name: str | None = None
    bound: dict[str, "ParsedFuncKind"] = field(default_factory=dict)
    owned: set[int] = field(default_factory=set)


_DECLARING: list[_Entry] = []


def enclosing_declaration(frame: FrameType | None) -> _Entry | None:
    """The active ``@module`` declaration whose class body encloses *frame*.

    Matched on the declaring frame rather than on one being open anywhere: an
    unrelated or stale open declaration must not make a standalone function
    look like a member of a class body it was never written in.
    """
    while frame is not None:
        if "__qualname__" in frame.f_locals:
            for entry in reversed(_DECLARING):
                if entry.frame is frame.f_back:
                    entry.owner_name = frame.f_locals["__qualname__"].rsplit(".", 1)[-1]
                    return entry
        elif frame.f_code.co_name == "<module>":
            return None
        frame = frame.f_back
    return None


def _retarget_module_calls(owner: str, functions, attached: dict) -> None:
    """Rebuild each marked call against the child attached under its binding.

    Runs before ``Module`` construction seals the functions. The record is
    repointed at the attached child before the rebuild reads it, because
    attaching may have copied the Module the class body named. A binding the
    class body does not attach is refused: there is no child to rebuild against,
    and collecting it would leave the call pointing outside the tree being built.
    """
    from tilefoundry.ir.core import (  # noqa: PLC0415 — avoid import cycle
        Expr,
        FunctionScope,
        TypeInferContext,
        get_metadata,
    )
    from tilefoundry.ir.core.expr import (  # noqa: PLC0415 — avoid import cycle
        Call,
    )
    from tilefoundry.ir.hir.function import Function as HirFunction  # noqa: PLC0415
    from tilefoundry.ir.hir.function import elaborate  # noqa: PLC0415
    from tilefoundry.ir.visitor import ExprWalker  # noqa: PLC0415
    from tilefoundry.parser.base import _ModuleCallee  # noqa: PLC0415

    unattached: list[str] = []

    class _RetargetVisitor(ExprWalker[None]):
        def visit(self, expr):
            if expr is None or not isinstance(expr, Expr):
                return None
            return super().visit(expr)

        def visit_Call(self, expr: Call) -> None:
            record = get_metadata(expr, _ModuleCallee)
            if isinstance(expr.target, HirFunction) and record is None:
                self.visit(expr.target)
            elif isinstance(expr.target, HirFunction) and record.binding not in attached:
                unattached.append(record.binding)
            elif isinstance(expr.target, HirFunction):
                child = attached[record.binding]
                entry = child.entry_function()
                object.__setattr__(
                    expr,
                    "metadata",
                    tuple(
                        _ModuleCallee(record.binding, child)
                        if isinstance(m, _ModuleCallee)
                        else m
                        for m in expr.metadata
                    ),
                )
                object.__setattr__(
                    expr,
                    "target",
                    elaborate(
                        entry,
                        tuple(a.type for a in expr.args),
                        TypeInferContext(scope=FunctionScope(child, entry)),
                        call=expr,
                    ),
                )
                object.__setattr__(
                    expr,
                    "metadata",
                    tuple(m for m in expr.metadata if not isinstance(m, _ModuleCallee)),
                )
            self.visit_operands(expr)

        def visit_Function(self, fn) -> None:
            self.visit_operands(fn)
            for variant in fn.variants:
                self.visit(variant)
            for converter in fn.converters:
                if isinstance(converter, tuple):
                    self.visit(converter[-1])

    visitor = _RetargetVisitor()
    for fn in functions:
        visitor.visit(fn)
    if unattached:
        raise ValueError(
            f"@module {owner!r}: call(s) to Module(s) {sorted(set(unattached))} that "
            f"no class-body binding attaches; it binds {sorted(attached)}. A Module "
            f"call is rebuilt against the child attached under the binding it names, "
            f"so a name nothing binds has no child to call"
        )


def _validate(topologies) -> tuple:
    from tilefoundry.ir.types.shard.mesh import Topology  # noqa: PLC0415

    if not isinstance(topologies, tuple) or not all(
        isinstance(topology, Topology) for topology in topologies
    ):
        raise TypeError(
            f"@module: topologies must be a tuple of Topology, got {topologies!r}"
        )
    return topologies


def module(
    cls=None, *, entry: str | None = None, target=None, topologies=UNDECLARED
):
    """Collect a class body into a ``Module``.

    Members may be DSL functions, child modules, or orchestration methods. See
    [parser §2.7](docs/spec/parser.md#27-module-authoring-surface).

    ``entry`` optionally names which collected function is the default step.

    ``target`` declares the hardware this execution domain runs on; only the
    outermost Module declares it and nested Modules inherit it. ``topologies``
    declares the ordered parallel-resource hierarchy; omitting it inherits the
    owning Module's hierarchy and ``()`` declares a topology-free Module.
    """
    from tilefoundry.ir.core.module import Module  # noqa: PLC0415 — avoid import cycle
    from tilefoundry.ir.hir.function import Function as HirFunction  # noqa: PLC0415
    from tilefoundry.ir.tir.prim_function import PrimFunction  # noqa: PLC0415
    from tilefoundry.target.base import target_instance  # noqa: PLC0415

    if target is not None:
        target_instance(target)
    resolved_target = target
    declared_topologies = None if topologies is UNDECLARED else _validate(topologies)
    mine = _Entry(declared_topologies, sys._getframe(1))
    _DECLARING.append(mine)

    def _wrap(cls_inner):
        for index, declaring in enumerate(_DECLARING):
            if declaring is mine:
                del _DECLARING[index:]
                break
        functions = []
        child_modules = []
        methods = {}
        attached: dict[str, Module] = {}
        for name, value in vars(cls_inner).items():
            if name == "__call__":


                raise TypeError(
                    f"@module {cls_inner.__name__!r}: a class-body __call__ has no "
                    f"effect -- Python resolves it on the type, not on the Module "
                    f"instance this builds. Name the method `forward`, which "
                    f"<module>(...) delegates to."
                )
            if name.startswith("__") and name.endswith("__"):
                continue
            if isinstance(value, Module):


                child = value if value.name == name else value.renamed(name)
                child_modules.append(child)
                attached[name] = child
                continue
            if isinstance(value, (tuple, list)) and value and all(
                isinstance(m, Module) for m in value
            ):

                child_modules.extend(value)
                continue
            if isinstance(value, (HirFunction, PrimFunction)):
                functions.append(value)
                continue
            if inspect.isfunction(value):
                methods[name] = value
                continue
            raise TypeError(
                f"@module {cls_inner.__name__!r}: member {name!r} is a "
                f"{type(value).__name__}, not an @func / @prim_func result, a "
                f"Module (or tuple/list of Modules), or a plain function; a "
                f"@module class body may contain only these three member kinds"
            )

        functions = [
            fn for fn in functions
            if id(fn) not in mine.owned and not getattr(fn, "specializations", ())
        ]
        if not functions and not child_modules and not methods:
            raise TypeError(
                f"@module {cls_inner.__name__!r}: empty class body; a Module must "
                f"declare at least one @func / @prim_func, child Module, or "
                f"orchestration method"
            )
        names = [fn.name for fn in functions]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(
                f"@module {cls_inner.__name__!r}: duplicate function name(s) "
                f"{dupes} (a class-body alias of a DSL function is not allowed; "
                f"one name maps to one function)"
            )
        mod_names = [m.name for m in child_modules]
        mod_dupes = sorted({n for n in mod_names if mod_names.count(n) > 1})
        if mod_dupes:
            raise ValueError(
                f"@module {cls_inner.__name__!r}: duplicate child module name(s) "
                f"{mod_dupes} (a class-body alias of a nested @module is not "
                f"allowed; one name maps to one child module)"
            )
        _retarget_module_calls(cls_inner.__name__, functions, attached)
        if entry is not None and entry not in names:
            raise ValueError(
                f"@module {cls_inner.__name__!r}: entry {entry!r} names no "
                f"collected function (have {names})"
            )
        return Module(
            name=cls_inner.__name__,
            functions=tuple(functions),
            entry=entry,
            modules=tuple(child_modules),
            target=resolved_target,
            topologies=declared_topologies,
            methods=methods,
        )

    if cls is not None:
        return _wrap(cls)
    return _wrap


__all__ = ["UNDECLARED", "module"]
