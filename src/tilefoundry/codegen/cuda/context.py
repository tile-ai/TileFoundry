"""What the shared codegen context does not know: CUDA's types and counters.

Everything a compile carries regardless of target lives in
:class:`tilefoundry.codegen.context.CodegenContext`; what is added here is
CUDA's alone -- the dtype spelling, the named-barrier ids the hardware counts,
and the device state a kernel asks for while it is being written. Handler
registration lives in ``tilefoundry.visitor_registry``, keyed by this target.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from tilefoundry.codegen.context import EmitContext
from tilefoundry.codegen.signature import CallableSignature, TensorSignature
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.target import CudaTarget
from tilefoundry.target.base import Target
from tilefoundry.visitor_registry.registries import codegen_registry

Geometry = tuple[tuple[object, object, object], tuple[object, object, object]]

_CUDA_CPP: dict[str, str] = {
    "f32": "float",
    "bf16": "__nv_bfloat16",
    "f16": "half",
    "fp8e4m3": "__nv_fp8_e4m3",
    "i32": "int",
    "i64": "long long",
}


def topology_scope_str(name: str) -> str:
    """Map a topology level name to its C++ ``tilefoundry::TopologyScope`` enumerator.

    Loud on an unknown level rather than silently defaulting.
    """
    scopes = {
        "gpu": "tilefoundry::TopologyScope::gpu",
        "cta": "tilefoundry::TopologyScope::cta",
        "thread": "tilefoundry::TopologyScope::thread",
    }
    try:
        return scopes[name]
    except KeyError:
        raise ValueError(
            f"unknown topology level {name!r}; expected one of {sorted(scopes)}"
        ) from None


class CudaCodegenContext(EmitContext):
    """A compile writing CUDA: the shared context plus what only CUDA states."""

    target_kind = CudaTarget

    def __init__(
        self,
        *,
        symbols: Mapping[int, CallableSignature] | None = None,
        target: Target | None = None,
        launches: Mapping[int, Geometry] | None = None,
        codegen_context: object | None = None,
    ) -> None:
        super().__init__(
            codegen_registry,
            symbols=symbols,
            target=target,
            codegen_context=codegen_context,
        )
        self._mesh_aliases: dict[int, tuple[str, str]] = {}
        self.launches: Mapping[int, Geometry] = {} if launches is None else launches
        """The geometry each device function is launched at, keyed by ``id(fn)``."""
        self.dynamic_extents: dict[str, str] = {}
        """Where the kernel being written reads each open dimension's extent."""
        self._next_barrier_id = 1
        self.needs_grid_barrier_state = False
        """Set while emitting a grid barrier, which the module declares state for."""
        self._has_smem_base = False

    def reset_smem_base(self) -> None:
        """Start a kernel with no dynamic shared-memory base declaration."""
        self._has_smem_base = False

    def smem_base(self) -> str:
        """Return the byte-addressed dynamic shared-memory base, declaring it once."""
        if not self._has_smem_base:
            self.emit("extern __shared__ unsigned char smem_base[];")
            self._has_smem_base = True
        return "smem_base"

    def bind_extents(self, params: Iterable[TensorSignature]) -> None:
        """Say where the kernel reads the extent of every dimension its types leave open.

        The type is the only record of an open dimension, so the extent beside
        the pointer is where a body reads it; the first parameter naming a
        dimension is the one it is read from.
        """
        for signature in params:
            for axis, dim in enumerate(signature.type.shape):
                if isinstance(dim, DimVar):
                    self.dynamic_extents.setdefault(dim.name, self.local_extent(signature, axis))

    def reset_barrier_ids(self) -> None:
        """Reset the named-barrier id counter at the start of a kernel body."""
        self._next_barrier_id = 1

    def alloc_barrier_id(self) -> int:
        """Allocate the next named-barrier id for a sub-CTA sync in this kernel.

        Hardware exposes ids 0..15; id 0 is reserved for the whole-CTA barrier,
        so 1..15 are available. Each emitted ``bar.sync`` draws a fresh id; a
        sync op node emits once, so a loop body reuses its id. Raises when a
        single kernel needs more distinct named barriers than the hardware has:
        wrapping would hand two live runs one id, and ``bar.sync`` counts
        arrivals per id, so the first to fill its own count would release the
        other's threads early.
        """
        bid = self._next_barrier_id
        if bid > 15:
            raise ValueError(
                "T.sync: too many distinct named barriers in one kernel "
                "(hardware supports ids 1..15 for sub-CTA sync)"
            )
        self._next_barrier_id = bid + 1
        return bid

    def dtype_to_cpp(self, dtype_name: str) -> str:
        t = _CUDA_CPP.get(dtype_name)
        if t is None:
            raise ValueError(f"unsupported dtype for CUDA codegen: {dtype_name!r}")
        return t


__all__ = ["CudaCodegenContext", "topology_scope_str"]
