"""``@module`` decorator — collect a class body into an IR ``Module``."""

from __future__ import annotations

import inspect
import sys

from tilefoundry import target as tilefoundry_target
from tilefoundry import verifier
from tilefoundry.interfaces import typeinfer
from tilefoundry.ir import core, types

from .ast_pattern import consume_module_context, create_module_context


class Undeclared:
    """Distinguish an omitted topology declaration from an explicit empty one.

    An explicit empty ``topologies`` declares a topology-free execution domain,
    while omitting it declares nothing and inherits from the owning Module. A
    plain ``None`` default cannot express both, because ``None`` is already the
    IR's encoding of "inherit". This applies to both declaration surfaces: the
    ``topologies=`` arguments of ``@module`` and standalone ``@func``.
    """

    def __repr__(self) -> str:
        return "UNDECLARED"


UNDECLARED = Undeclared()


def validate_topologies(topologies) -> tuple:
    if not isinstance(topologies, tuple) or not all(
        isinstance(topology, types.Topology) for topology in topologies
    ):
        raise TypeError(
            f"@module: topologies must be a tuple of Topology, got {topologies!r}"
        )
    return topologies


def collect_module(cls, *, entry, target, topologies) -> core.Module:
    """Collect the members of the class body *cls* into a ``Module``."""
    functions = []
    children = {}
    methods = {}
    for name, value in vars(cls).items():
        if name == "__call__":
            raise TypeError(
                f"@module {cls.__name__!r}: a class-body __call__ has no "
                f"effect -- Python resolves it on the type, not on the Module "
                f"instance this builds. Name the method `forward`, which "
                f"<module>(...) delegates to."
            )
        if name.startswith("__") and name.endswith("__"):
            continue
        if isinstance(value, (core.Module, core.ModuleList)):
            slot = core.ModuleList(value) if isinstance(value, core.Module) else value
            slot.bind(name)
            children[name] = slot
            continue
        if isinstance(value, (tuple, list)) and value and all(
            isinstance(m, core.Module) for m in value
        ):
            raise TypeError(
                f"@module {cls.__name__!r}: member {name!r} is a list/tuple of Modules; "
                "use ModuleList(definition, N) instead"
            )
        if isinstance(value, (core.Function, core.PrimFunction)):
            functions.append(value)
            continue
        if inspect.isfunction(value):
            methods[name] = value
            continue
        raise TypeError(
            f"@module {cls.__name__!r}: member {name!r} is a "
            f"{type(value).__name__}, not an @func / @prim_func result, a "
            f"Module / ModuleList, or a plain function; a "
            f"@module class body may contain only these three member kinds"
        )

    functions = [fn for fn in functions if not getattr(fn, "specializations", ())]
    if not functions and not children and not methods:
        raise TypeError(
            f"@module {cls.__name__!r}: empty class body; a Module must "
            f"declare at least one @func / @prim_func, child Module, or "
            f"orchestration method"
        )
    names = [fn.name for fn in functions]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"@module {cls.__name__!r}: duplicate function name(s) "
            f"{dupes} (a class-body alias of a DSL function is not allowed; "
            f"one name maps to one function)"
        )
    if entry is not None and entry not in names:
        raise ValueError(
            f"@module {cls.__name__!r}: entry {entry!r} names no "
            f"collected function (have {names})"
        )
    return core.Module(
        name=cls.__name__,
        functions=tuple(functions),
        entry=entry,
        children=children,
        target=target,
        topologies=topologies,
        methods=methods,
    )


def module(
    cls=None, *, entry: str | None = None, target=None, topologies=UNDECLARED
):
    """Collect a class body into a ``Module``.

    Members may be DSL functions, child modules, or orchestration methods. See
    [parser §3](docs/spec/parser.md#3-implementation-overview).

    ``entry`` optionally names which collected function is the default step.

    ``target`` declares the hardware this execution domain runs on. A child
    with no declaration inherits its caller's target. ``topologies``
    declares the ordered parallel-resource hierarchy; omitting it inherits the
    owning Module's hierarchy and ``()`` declares a topology-free Module.
    """
    if target is not None:
        tilefoundry_target.instance(target)
    declared_topologies = None if topologies is UNDECLARED else validate_topologies(topologies)
    if cls is not None:
        return collect_module(
            cls, entry=entry, target=target, topologies=declared_topologies
        )
    owner_frame = sys._getframe(1)
    context = create_module_context(
        entry=entry,
        target=target,
        topologies=declared_topologies,
        owner_frame=owner_frame,
        source_filename=owner_frame.f_code.co_filename,
    )

    def wrap_with_context(cls_inner):
        try:
            result = context.finalize(cls_inner)
            for function in result.functions:
                if isinstance(function, core.Function):
                    typeinfer.infer_function(
                        function,
                        typeinfer.TypeInferContext(module=result, function=function),
                    )
                    verifier.verify_function(function, module=result)
                elif isinstance(function, core.PrimFunction):
                    verifier.verify_prim_function(function, module_fns=result)
            return result
        except Exception:
            consume_module_context(context)
            raise

    return wrap_with_context


__all__ = ["UNDECLARED", "module"]
