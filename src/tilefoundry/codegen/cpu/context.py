"""What the shared codegen context does not know: how a host entry holds a tensor.

A host entry is handed runtime tensors, not pointers, so what it passes on for
a parameter and where it reads an extent are both reached through the tensor
object. Everything a compile carries regardless of target lives in
:class:`tilefoundry.codegen.Context`.
"""

from __future__ import annotations

from collections.abc import Mapping

from tilefoundry import target as tilefoundry_target
from tilefoundry.codegen import context, registry, signature
from tilefoundry.target import cpu


@tilefoundry_target.bind(cpu.Target, context.EmitContext)
class Context(context.EmitContext):
    """A compile writing the host unit: the shared context plus what a tensor is here."""

    target_kind = cpu.Target

    @classmethod
    def create(cls, target, view):
        """Create target-owned emission state from a group view."""
        return cls(
            symbols=view.symbols, target=target, resolved_launches=view.resolved_launches,
            codegen_context=view.codegen,
        )

    def __init__(
        self,
        *,
        symbols: Mapping[int, signature.CallableSignature] | None = None,
        target: tilefoundry_target.Target | None = None,
        resolved_launches: Mapping[int, object] | None = None,
        codegen_context: object | None = None,
    ) -> None:
        super().__init__(
            registry.registry,
            symbols=symbols,
            target=target,
            resolved_launches=resolved_launches,
            codegen_context=codegen_context,
        )

    def local_value(self, sig: signature.Signature) -> str:
        """The buffer behind a runtime tensor, which is what a device call takes."""
        if isinstance(sig, signature.TensorSignature):
            return f"{sig.name}.data_ptr()"
        return sig.name

    def local_extent(self, sig: signature.TensorSignature, axis: int) -> str:
        """An extent read off the tensor that carries it, rather than off a parameter."""
        return f"static_cast<long long>({sig.name}.shape()[{axis}])"


__all__ = ["Context"]
