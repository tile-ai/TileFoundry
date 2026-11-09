"""Define callable IR operations from ``ParamDef`` descriptors.

``parameters_of`` reflects descriptors in MRO order; construction accepts only
attribute parameters. Registration is explicit through ``@register_op`` and
the ordered ``OpSchema`` registry.

See [core-ir §2.3](docs/spec/core-ir.md#23-op).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tilefoundry import services

from .op_registry import iter_schemas, own_schema_of
from .param_def import ParamDef, ParamKind, collect_param_defs
from .register import get_registered_name
from .storage import resolve_storage

__all__ = [
    "Op",
    "OpCapability",
    "ParameterInfo",
    "capabilities_of",
    "parameters_of",
    "signature_of",
    "supported_op_capabilities",
    "target_capabilities",
]


def signature_of(cls: type) -> tuple[ParamDef, ...]:
    """The ``ParamDef`` tuple for ``cls``.

    The registered ``OpSchema``'s ``signature`` when ``@register_op`` has run,
    else a fresh reflection walk (e.g. an unregistered test fixture subclassing
    ``Op`` directly).
    """
    schema = own_schema_of(cls)
    if schema is not None:
        return schema.signature
    return collect_param_defs(cls)


def normalize_attr(name: str, value: Any) -> Any:
    """Normalise known typed attributes at the IR construction boundary.

    A ``storage`` attribute is coerced from a legacy string alias to
    ``StorageKind | None`` so an Op instance never carries a raw string.
    """
    if name == "storage":
        return resolve_storage(value)
    return services.normalize_dim(value)


@dataclass(frozen=True)
class ParameterInfo:
    """Lightweight reflection record for a declared Op parameter."""

    name: str
    kind: ParamKind
    type: Any


@dataclass(frozen=True)
class OpCapability:
    """One target capability exposed by a registered instruction Op.

    ``name=None`` imposes no named capability requirement; the target must still
    satisfy the instruction's execution mesh Pattern at each call. A carrier
    with several concrete declarations lists one record per variant; the
    declaration and attribute say how an instance of that declaration is bound
    back to the carrier without teaching registry consumers about its shape.
    """

    name: str | None
    declaration: type | None = None
    attribute: str | None = None

    def __post_init__(self) -> None:
        if (self.declaration is None) != (self.attribute is None):
            raise ValueError("a capability variant states both declaration and attribute")


def capabilities_of(op_type: type) -> tuple[OpCapability, ...]:
    """Return the normalized target-admission declarations of an Op."""
    stated = vars(op_type).get("capability")
    if isinstance(stated, OpCapability):
        return (stated,)
    if (
        isinstance(stated, tuple)
        and stated
        and all(isinstance(item, OpCapability) for item in stated)
    ):
        return stated
    if stated is None:
        return ()
    raise ValueError(f"{get_registered_name(op_type)} has an invalid op capability declaration")


def target_capabilities(target: object) -> frozenset[str]:
    """The instruction capabilities ``target``'s architecture lists; none without one."""
    architecture = getattr(target, "architecture", None)
    return frozenset(getattr(architecture, "capabilities", ()))

def supported_op_capabilities(target: object) -> tuple[tuple[type, OpCapability], ...]:
    """Enumerate TIR variants admitted by target capabilities."""
    supported = target_capabilities(target)
    return tuple(
        (schema.op_class, capability)
        for schema in iter_schemas()
        if schema.dialect == "T" and schema.op_class is not None
        for capability in capabilities_of(schema.op_class)
        if (capability.name is None or capability.name in supported)
    )


class ParameterCache:
    """The ``ParameterInfo`` list of every Op class asked about so far."""

    def __init__(self) -> None:
        """Start empty."""
        self.entries: dict[type, list[ParameterInfo]] = {}

    def of(self, cls: type) -> list[ParameterInfo]:
        """The ``ParameterInfo`` projection of ``cls``'s ``ParamDef`` signature."""
        cached = self.entries.get(cls)
        if cached is not None:
            return cached
        infos = [
            ParameterInfo(name=pd.name, kind=pd.kind, type=pd.annotation)
            for pd in signature_of(cls)
        ]
        self.entries[cls] = infos
        return infos


parameter_cache = ParameterCache()


def parameters_of(cls: type) -> list[ParameterInfo]:
    """``ParameterInfo`` projection of ``cls``'s ``ParamDef`` signature.

    Uses the schema signature attached by ``@register_op`` when present
    (``signature_of``); otherwise reflects ``ParamDef`` class-body descriptors
    directly (base→derived MRO order, derived redeclaration overrides in place).
    """
    return parameter_cache.of(cls)


class Op:
    """All Op classes inherit from this. Reflection-based param discovery."""

    def __new__(cls, **attrs: Any):
        """The one instance of an Op with no attribute, or a fresh instance."""
        if not attrs:
            attr_params = [p for p in parameters_of(cls) if p.kind == "attribute"]
            if not attr_params:
                cached = cls.__dict__.get("singleton")
                if cached is not None:
                    return cached
                inst = super().__new__(cls)
                cls.singleton = inst
                return inst
        return super().__new__(cls)

    def __init__(self, **attrs: Any) -> None:
        """Set the attributes the Op declares; refuse unknown and missing ones."""
        param_defs = signature_of(type(self))
        attr_defs = {pd.name: pd for pd in param_defs if pd.kind == "attribute"}
        for k, v in attrs.items():
            if k not in attr_defs:
                raise TypeError(f"{type(self).__name__}: unknown attribute {k!r}")
            setattr(self, k, normalize_attr(k, v))

        missing = set(attr_defs) - set(attrs)
        for m in list(missing):
            pd = attr_defs[m]
            if pd.has_default:
                setattr(self, m, normalize_attr(m, pd.default))
                missing.discard(m)
        if missing:
            raise TypeError(f"{type(self).__name__}: missing attribute(s) {sorted(missing)}")

    def __repr__(self) -> str:
        """The class name and its attributes."""
        infos = parameters_of(type(self))
        attr_bits = [
            f"{p.name}={getattr(self, p.name, '?')!r}" for p in infos if p.kind == "attribute"
        ]
        return f"{type(self).__name__}({', '.join(attr_bits)})"


