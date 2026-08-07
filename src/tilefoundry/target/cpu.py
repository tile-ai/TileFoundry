"""CPU compilation target."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.target.base import Target, register_target
from tilefoundry.target.services import CodeGenerator


@register_target
@dataclass(frozen=True)
class CpuTarget(Target):
    """Identify the CPU host backend."""

    name = "cpu"

    def get_code_generator(self) -> CodeGenerator:
        from tilefoundry.codegen.cpu.module import CPU_CODE_GENERATOR  # noqa: PLC0415

        return CPU_CODE_GENERATOR


__all__ = ["CpuTarget"]
