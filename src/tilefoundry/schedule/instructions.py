"""Reflect target-supported instruction declarations from the Op registry."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import InstructionCapability
from tilefoundry.ir.core.op_registry import iter_schemas
from tilefoundry.target import Target


def _op_id(op_type: type) -> str:
    schema = op_type._op_schema
    return f"{schema.dialect}.{schema.name}"


def _declaration_id(declaration: type) -> str:
    return getattr(declaration, "reference_name", "") or _op_id(declaration)


@dataclass(frozen=True)
class Instruction:
    """One concrete declaration behind a registered instruction carrier."""

    op_type: type
    declaration: type
    capability: str | None
    report_order: int
    attribute: str | None = None

    @property
    def id(self) -> str:
        return _declaration_id(self.declaration)

    @property
    def is_variant(self) -> bool:
        return self.attribute is not None

    def instantiate(self, variant=None):
        """Build the carrier, binding a declared variant when it has one."""
        if self.attribute is None:
            if variant is not None:
                raise ValueError(f"{self.id} is not a variant instruction")
            return self.op_type()
        if variant is None:
            raise ValueError(f"{self.id} requires a declaration instance")
        return self.op_type(**{self.attribute: variant})


@dataclass(frozen=True)
class InstructionFamily:
    """One registered Op and the declarations admitted by a target."""

    op_type: type
    declarations: tuple[Instruction, ...]

    @property
    def id(self) -> str:
        return _op_id(self.op_type)

    @property
    def capability(self) -> str | tuple[str, ...] | None:
        names = tuple(
            declaration.capability
            for declaration in self.declarations
            if declaration.capability is not None
        )
        if not names:
            return None
        return names[0] if len(names) == 1 else names

    @property
    def report_order(self) -> int:
        return min(declaration.report_order for declaration in self.declarations)


def _declared_capabilities(op_type: type) -> tuple[InstructionCapability, ...] | None:
    stated = vars(op_type).get("capability")
    if stated is None:
        return None
    if isinstance(stated, InstructionCapability):
        return (stated,)
    if isinstance(stated, tuple) and stated and all(
        isinstance(item, InstructionCapability) for item in stated
    ):
        return stated
    schema = op_type._op_schema
    raise ValueError(
        f"{schema.dialect}.{schema.name} has an invalid instruction capability declaration"
    )


def _supported(name: str | None, target: Target) -> bool:
    if name is None:
        return True
    architecture = getattr(target, "architecture", None)
    return name in frozenset(getattr(architecture, "capabilities", ()))


def families(target: Target) -> tuple[InstructionFamily, ...]:
    """Return supported registered instruction Ops in stable report order."""
    declared = []
    for position, schema in enumerate(iter_schemas()):
        if schema.dialect != "T" or schema.op_class is None:
            continue
        capabilities = _declared_capabilities(schema.op_class)
        if capabilities is None:
            continue
        variants = tuple(
            Instruction(
                op_type=schema.op_class,
                declaration=capability.declaration or schema.op_class,
                capability=capability.name,
                report_order=capability.report_order,
                attribute=capability.attribute,
            )
            for capability in capabilities
            if _supported(capability.name, target)
        )
        if variants:
            declared.append((InstructionFamily(schema.op_class, variants), position))
    declared.sort(key=lambda item: (item[0].report_order, item[1]))
    return tuple(family for family, _position in declared)


def declarations(target: Target) -> tuple[Instruction, ...]:
    """Flatten supported families to the declarations candidate matching uses."""
    return tuple(declaration for family in families(target) for declaration in family.declarations)


__all__ = ["Instruction", "InstructionFamily", "declarations", "families"]
