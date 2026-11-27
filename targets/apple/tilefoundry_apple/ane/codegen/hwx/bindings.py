"""Named I/O surfaces of an HWX image: role, virtual address and loader-bound reservation."""

from dataclasses import dataclass

from tilefoundry_apple.ane import isa
from tilefoundry_apple.ane.isa import arith

from .macho import PAGE, SG_FVMLIB, SG_NORELOC, Section, Segment

BINDING_ROLE_INPUT = 1
BINDING_ROLE_OUTPUT = 2


@dataclass(frozen=True)
class TensorBinding:
    """One named tensor buffer with its role, virtual address and section."""

    name: str
    desc: isa.TensorDesc
    role: int = 0
    address: int = 0

    @property
    def size(self) -> int:
        """Page-aligned reservation of this I/O binding."""
        return arith.align_up(self.desc.storage_bytes, PAGE)

    def validate(self, layout: str = "canonical") -> None:
        """Check the binding symbol and its tensor geometry."""
        if not self.name or not self.name.isascii() or not self.name.isprintable():
            raise ValueError(f"invalid I/O symbol {self.name!r}")
        try:
            self.desc.validate(layout)
        except ValueError as error:
            raise ValueError(f"{self.name}: {error}") from error

    @property
    def tensor_symbol(self) -> str:
        """The label recorded in the binding record."""
        if self.role == BINDING_ROLE_OUTPUT:
            return f"{self.name}@output"
        return self.name

    def segment(self) -> Segment:
        """Reserve the surface for the loader: an output as ``__data``, an input as ``__const``."""
        output = self.role == BINDING_ROLE_OUTPUT
        name, protection, kind = (b"__data", 2, 0x23) if output else (b"__const", 1, 0x21)
        section = Section(name, self.address, self.desc.storage_bytes, PAGE, kind)
        return Segment(
            b"__FVMLIB", self.address, self.size, protection, (section,), SG_FVMLIB | SG_NORELOC
        )


__all__ = ["BINDING_ROLE_INPUT", "BINDING_ROLE_OUTPUT", "TensorBinding"]
