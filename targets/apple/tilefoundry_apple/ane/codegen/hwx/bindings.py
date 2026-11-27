"""H14 HWX tensor binding encoding."""

from dataclasses import dataclass

from tilefoundry_apple.ane import isa
from tilefoundry_apple.ane.isa import arith

from .macho import (
    PAGE,
    SECT_ALIGN_IO,
    SECT_CONST_FLAGS,
    SECT_DATA_FLAGS,
    SEG_FVMLIB_FLAGS,
    Section,
    Segment,
)

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
    def section_name(self) -> bytes:
        """The __FVMLIB section that reserves this surface."""
        return b"__data" if self.role == BINDING_ROLE_OUTPUT else b"__const"

    @property
    def tensor_symbol(self) -> str:
        """The label recorded in the binding record."""
        if self.role == BINDING_ROLE_OUTPUT:
            return f"{self.name}@output"
        return self.name


def binding_segment(binding: TensorBinding) -> Segment:
    """Reserve one loader-bound I/O section with its role's protection bits."""
    protection = 2 if binding.role == BINDING_ROLE_OUTPUT else 1
    return Segment(
        b"__FVMLIB",
        binding.address,
        binding.size,
        0,
        0,
        protection,
        protection,
        SEG_FVMLIB_FLAGS,
        (
            Section(
                binding.section_name,
                b"__FVMLIB",
                binding.address,
                binding.desc.storage_bytes,
                0,
                SECT_ALIGN_IO,
                SECT_DATA_FLAGS if binding.role == BINDING_ROLE_OUTPUT else SECT_CONST_FLAGS,
            ),
        ),
    )


__all__ = ["BINDING_ROLE_INPUT", "BINDING_ROLE_OUTPUT", "TensorBinding", "binding_segment"]
