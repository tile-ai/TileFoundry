"""What the shared codegen context does not know: how a host entry holds a tensor.

A host entry is handed runtime tensors, not pointers, so what it passes on for
a parameter and where it reads an extent are both reached through the tensor
object. Everything a compile carries regardless of target lives in
:class:`tilefoundry.codegen.context.CodegenContext`.
"""

from __future__ import annotations

from collections.abc import Mapping

from tilefoundry.codegen.context import CodegenContext
from tilefoundry.codegen.signature import CallableSignature, Signature, TensorSignature
from tilefoundry.target import CpuTarget
from tilefoundry.target.base import Target
from tilefoundry.visitor_registry.registries import codegen_registry


class CpuCodegenContext(CodegenContext):
    """A compile writing the host unit: the shared context plus what a tensor is here."""

    target_kind = CpuTarget

    def __init__(
        self,
        *,
        symbols: Mapping[int, CallableSignature] | None = None,
        target: Target | None = None,
    ) -> None:
        super().__init__(codegen_registry, symbols=symbols, target=target)

    def local_value(self, signature: Signature) -> str:
        """The buffer behind a runtime tensor, which is what a device call takes."""
        if isinstance(signature, TensorSignature):
            return f"{signature.name}.data_ptr()"
        return signature.name

    def local_extent(self, signature: TensorSignature, axis: int) -> str:
        """An extent read off the tensor that carries it, rather than off a parameter."""
        return f"static_cast<long long>({signature.name}.shape()[{axis}])"


__all__ = ["CpuCodegenContext"]
