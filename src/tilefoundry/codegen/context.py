"""The context one compile writes through, whatever targets it writes for.

It spans the whole compile rather than one translation unit: the symbol table
is read by every emitter, and the boundary of the unit being written is drawn
by ``capture`` instead of by a second object. A target subclasses this to add
what only it knows -- its type names, its hardware counters, and how its own
scope spells a value it hands to someone else.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from tilefoundry import dispatch
from tilefoundry import target as tilefoundry_target
from tilefoundry.ir import core, tir

from .context_builder import Context
from .emitter import CppEmitter
from .registry import Role, registry
from .signature import CallableSignature, Signature, TensorSignature


class EmitContext:
    """One compile, from the first function read to the last line written."""

    target_kind: type[tilefoundry_target.Target]
    """Whose handlers answer here; each target's subclass states its own."""

    exported: bool
    """Whether the parameter list being written is one another unit will read.

    Only what plain C spells crosses that line, so a target with types of its
    own answers with the C ABI while this is set.
    """

    def __init__(
        self,
        registry: dispatch.Registry,
        *,
        symbols: Mapping[int, CallableSignature] | None = None,
        target: tilefoundry_target.Target | None = None,
        resolved_launches: Mapping[int, object] | None = None,
        codegen_context: Context | None = None,
        target_kind: type[tilefoundry_target.Target] | None = None,
    ) -> None:
        self.registry = registry
        self.codegen_context = codegen_context
        self.symbols: Mapping[int, CallableSignature] = {} if symbols is None else symbols
        self.target = target
        self.resolved_launches = {} if resolved_launches is None else resolved_launches
        self.exported = False
        if target_kind is not None:
            self.target_kind = target_kind
        self.lines: list[str] = []
        self.depth = 0
        self.var_names: dict[int, str] = {}
        self.counter = 0
        self.kernel_param_ids: set[int] = set()

    @classmethod
    def create(cls, target, view) -> EmitContext:
        """Create generic emission state from the compile-wide group view."""
        return cls(
            registry, symbols=view.symbols, target=target,
            resolved_launches=view.resolved_launches, codegen_context=view.codegen,
            target_kind=type(target),
        )

    def callee_of(self, launch):
        """The callee root context resolved for one Launch statement."""
        callee = self.resolved_launches.get(id(launch))
        if callee is None:
            raise KeyError("codegen: Launch was not resolved by the root context")
        return callee

    def signature_of(self, fn) -> CallableSignature:
        """How a caller writes a call to *fn*, as the symbol table settled it."""
        signature = self.symbols.get(id(fn))
        if signature is None:
            raise KeyError(
                f"codegen: {getattr(fn, 'name', fn)!r} is not in this compile's "
                f"symbol table, so nothing states how to call it"
            )
        return signature

    def register_kernel_param(self, var) -> None:
        """Bind a declared parameter to its own name, which no counter suffixes."""
        key = id(var)
        self.var_names[key] = var.name
        self.kernel_param_ids.add(key)

    def is_kernel_param(self, var) -> bool:
        return id(var) in self.kernel_param_ids

    def emit(self, line: str) -> None:
        self.lines.append("  " * self.depth + line)

    def blank(self) -> None:
        self.lines.append("")

    def indent(self) -> None:
        self.depth += 1

    def dedent(self) -> None:
        self.depth -= 1

    def next_id(self) -> int:
        """The next number of the counter that suffixes every invented name."""
        self.counter += 1
        return self.counter

    def name_for(self, var) -> str:
        """The C++ name of *var*, invented once and kept for the rest of the compile."""
        key = id(var)
        if key in self.var_names:
            return self.var_names[key]
        name = f"{var.name}_{self.next_id()}"
        self.var_names[key] = name
        return name

    def source(self) -> str:
        return "\n".join(self.lines) + "\n"

    def capture(self, fn) -> str:
        """Run ``fn(ctx)`` against a fresh line buffer and return what it wrote.

        Indentation and names carry across the swap; only the buffer is
        isolated, which is how a sub-sequence becomes a string a template can
        splice in.
        """
        saved_lines = self.lines
        self.lines = []
        try:
            fn(self)
            return "\n".join(self.lines)
        finally:
            self.lines = saved_lines

    def emit_node(self, node) -> None:
        """Write *node* through the target Visitor or function orchestrator."""
        if isinstance(
            node, (tir.abort.Abort, core.Evaluate, core.For, core.If, core.LetStmt, core.MeshScope, core.Return, core.Sequential, core.While)
        ):
            CppEmitter(context=self).visit(node)
            return
        self.handler_for(type(node))(node, self)

    def stmt_handler(self, node_type: type) -> Callable | None:
        """Return the optional target-specific statement writer."""
        return self.registry.get((self.target_kind, Role.EMIT, node_type), None)

    def handler_for(self, node_type: type) -> Callable:
        """The handler that writes *node_type* here, for a caller that dispatches itself."""
        handler = self.stmt_handler(node_type)
        if handler is None:
            raise RuntimeError(
                f"no codegen handler for {node_type.__name__} on {tilefoundry_target.summary(self.target)}"
            )
        return handler

    def declare(
        self,
        signature: Signature,
        *,
        exported: bool = False,
        by: type | None = None,
    ) -> tuple[str, ...]:
        """The C++ parameters the target *by* (default: this context's) declares for a signature."""
        handler = self.lookup((by or self.target_kind, Role.CALLEE, type(signature)))
        outer, self.exported = self.exported, exported
        try:
            return handler(signature, self)
        finally:
            self.exported = outer

    def parameters(
        self,
        signature: CallableSignature,
        *,
        exported: bool = False,
        by: type | None = None,
    ) -> str:
        """The whole parameter list of one call, in call order."""
        return ", ".join(
            token
            for param in signature.all_params
            for token in self.declare(param, exported=exported, by=by)
        )

    def local_value(self, signature: Signature) -> str:
        """What this scope calls the value it passes for *signature*.

        The default is a local of the parameter's own name, which is what a
        function that received it has; an entry that holds something richer
        than the parameter says so by overriding this.
        """
        return signature.name

    def local_extent(self, signature: TensorSignature, axis: int) -> str:
        """What this scope calls *signature*'s extent along *axis*.

        A plain C parameter list carries each open extent beside the pointer
        under the name the declaration gave it; an entry holding a runtime
        tensor reads the extent off that tensor instead.
        """
        return signature.extent_name(axis)

    def lookup(self, key) -> Callable:
        handler = self.registry.get(key, None)
        if handler is None:
            raise RuntimeError(f"{self.registry.name}: nothing registered for {dispatch.spelled(key)}")
        return handler


__all__ = ["EmitContext"]
