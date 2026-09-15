"""The context one compile writes through, whatever targets it writes for.

It spans the whole compile rather than one translation unit: the symbol table
is read by every emitter, and the boundary of the unit being written is drawn
by ``capture`` instead of by a second object. A target subclasses this to add
what only it knows -- its type names, its hardware counters, and how its own
scope spells a value it hands to someone else.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from tilefoundry.codegen.signature import CallableSignature, Signature, TensorSignature
from tilefoundry.ir.tir.stmts import Evaluate
from tilefoundry.target.base import Target
from tilefoundry.visitor_registry.registries import DispatchRegistry, Role, spelled


class EmitContext:
    """One compile, from the first function read to the last line written."""

    target_kind: type[Target]
    """Whose handlers answer here; each target's subclass states its own."""

    exported: bool
    """Whether the parameter list being written is one another unit will read.

    Only what plain C spells crosses that line, so a target with types of its
    own answers with the C ABI while this is set.
    """

    def __init__(
        self,
        registry: DispatchRegistry,
        *,
        symbols: Mapping[int, CallableSignature] | None = None,
        target: Target | None = None,
        resolved_launches: Mapping[int, object] | None = None,
        codegen_context: object | None = None,
    ) -> None:
        self.registry = registry
        self.codegen_context = codegen_context
        self.symbols: Mapping[int, CallableSignature] = {} if symbols is None else symbols
        self.target = target
        self.resolved_launches = {} if resolved_launches is None else resolved_launches
        self.exported = False
        self._lines: list[str] = []
        self._indent = 0
        self._var_names: dict[int, str] = {}
        self._counter = 0
        self._kernel_param_ids: set[int] = set()

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
        self._var_names[key] = var.name
        self._kernel_param_ids.add(key)

    def is_kernel_param(self, var) -> bool:
        return id(var) in self._kernel_param_ids

    def emit(self, line: str) -> None:
        self._lines.append("  " * self._indent + line)

    def blank(self) -> None:
        self._lines.append("")

    def indent(self) -> None:
        self._indent += 1

    def dedent(self) -> None:
        self._indent -= 1

    def name_for(self, var) -> str:
        """The C++ name of *var*, invented once and kept for the rest of the compile."""
        key = id(var)
        if key in self._var_names:
            return self._var_names[key]
        self._counter += 1
        name = f"{var.name}_{self._counter}"
        self._var_names[key] = name
        return name

    def source(self) -> str:
        return "\n".join(self._lines) + "\n"

    def capture(self, fn) -> str:
        """Run ``fn(ctx)`` against a fresh line buffer and return what it wrote.

        Indentation and names carry across the swap; only the buffer is
        isolated, which is how a sub-sequence becomes a string a template can
        splice in.
        """
        saved_lines = self._lines
        self._lines = []
        try:
            fn(self)
            return "\n".join(self._lines)
        finally:
            self._lines = saved_lines

    def emit_node(self, node) -> None:
        """Write *node* through the target Visitor or function orchestrator."""
        from tilefoundry.ir.tir.abort import Abort  # noqa: PLC0415
        from tilefoundry.ir.tir.stmts import (  # noqa: PLC0415
            For,
            If,
            LetStmt,
            MeshScope,
            Return,
            Sequential,
            While,
        )

        if isinstance(
            node, (Abort, Evaluate, For, If, LetStmt, MeshScope, Return, Sequential, While)
        ):
            from tilefoundry.codegen.emitter import CudaEmitter  # noqa: PLC0415

            CudaEmitter(context=self).visit(node)
            return
        self.handler_for(type(node))(node, self)

    def handler_for(self, node_type: type) -> Callable:
        """The handler that writes *node_type* here, for a caller that dispatches itself."""
        return self._handler((self.target_kind, Role.EMIT, node_type))

    def declare(
        self,
        signature: Signature,
        callee_target: Target | None = None,
        *,
        exported: bool = False,
    ) -> tuple[str, ...]:
        """The C++ parameters *signature* contributes where *callee_target* declares it.

        One signature is one logical parameter; how many parameters that is in
        C++ is the declaring target's answer, and a derived name it invents
        there is never read back anywhere else. *callee_target* defaults to
        this context's own, so declaring a foreign symbol and defining it reach
        the one handler and cannot come to disagree.
        """
        kind = self.target_kind if callee_target is None else type(callee_target)
        handler = self._handler((kind, Role.CALLEE, type(signature)))
        outer, self.exported = self.exported, exported
        try:
            return handler(signature, self)
        finally:
            self.exported = outer

    def parameters(
        self,
        signature: CallableSignature,
        callee_target: Target | None = None,
        *,
        exported: bool = False,
    ) -> str:
        """The whole parameter list of one call, in call order."""
        return ", ".join(
            token
            for param in signature.all_params
            for token in self.declare(param, callee_target, exported=exported)
        )

    def argument(self, signature: Signature, callee_target: Target) -> tuple[str, ...]:
        """What this scope passes for *signature* in a call to a *callee_target* function.

        The callee's target answers, because how a call to it is written is its
        convention; this scope fills in only names it already holds. That is
        what lets a host entry call a device function it knows nothing about.
        """
        key = (type(callee_target), Role.CALLER, type(signature))
        return self._handler(key)(signature, self)

    def arguments(self, signature: CallableSignature, callee_target: Target) -> str:
        """The whole argument list of one call, in the order the callee declared."""
        return ", ".join(
            token for param in signature.all_params for token in self.argument(param, callee_target)
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

    def _handler(self, key) -> Callable:
        handler = self.registry.lookup(key)
        if handler is None:
            raise RuntimeError(f"{self.registry.name}: nothing registered for {spelled(key)}")
        return handler


__all__ = ["EmitContext"]
