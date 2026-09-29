"""Define callable IR operations from ``ParamDef`` descriptors.

``Op.params()`` reflects descriptors in MRO order; construction accepts only
attribute parameters. Registration is explicit through ``@register_op`` and
the ordered ``OpSchema`` registry.

See [core-ir §2.3](docs/spec/core-ir.md#23-op).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from tilefoundry.ir.core.param_def import ParamDef, _ParamKind, collect_param_defs
from tilefoundry.ir.types.storage import resolve_storage


def _signature(cls: type) -> tuple[ParamDef, ...]:
    """Signature.

    The ``ParamDef`` tuple for ``cls``: the attached ``OpSchema``'s
    ``signature`` when ``@register_op`` has run, else a fresh reflection
    walk (e.g. an unregistered test fixture subclassing ``Op`` directly).
    """
    schema = getattr(cls, "_op_schema", None)
    if schema is not None:
        return schema.signature
    return collect_param_defs(cls)


def _normalize_attr(name: str, value: Any) -> Any:
    """Normalise known typed attributes at the IR construction boundary.

    A ``storage`` attribute is coerced from a legacy string alias to
    ``StorageKind | None`` so an Op instance never carries a raw string.
    """
    if name == "storage":
        return resolve_storage(value)
    from tilefoundry.ir.isl_interop import normalize_dim_entries  # noqa: PLC0415

    return normalize_dim_entries(value)


@dataclass(frozen=True)
class ParameterInfo:
    """Lightweight reflection record for a declared Op parameter."""

    name: str
    kind: _ParamKind
    type: Any


@dataclass(frozen=True)
class OpCapability:
    """One target capability exposed by a registered instruction Op.

    ``name=None`` means that every target admits the instruction.  A carrier
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


def op_identifier(op_type: type) -> str:
    """Return the registered or externally declared name of an Op."""
    reference = getattr(op_type, "reference_name", "")
    if reference:
        return reference
    schema = op_type._op_schema
    return f"{schema.dialect}.{schema.name}"


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
    raise ValueError(f"{op_identifier(op_type)} has an invalid op capability declaration")


def supported_op_capabilities(target: object) -> tuple[tuple[type, OpCapability], ...]:
    """Enumerate registered TIR Ops admitted by ``target`` in registry order."""
    from tilefoundry.ir.core.op_registry import iter_schemas  # noqa: PLC0415

    architecture = getattr(target, "architecture", None)
    supported = frozenset(getattr(architecture, "capabilities", ()))
    return tuple(
        (schema.op_class, capability)
        for schema in iter_schemas()
        if schema.dialect == "T" and schema.op_class is not None
        for capability in capabilities_of(schema.op_class)
        if capability.name is None or capability.name in supported
    )


class Op:
    """All Op classes inherit from this. Reflection-based param discovery."""

    _params_cache: ClassVar[dict[type, list[ParameterInfo]]] = {}

    def __new__(cls, **attrs: Any):

        if not attrs:
            attr_params = [p for p in cls.params() if p.kind == "attribute"]
            if not attr_params:
                cached = cls.__dict__.get("_singleton")
                if cached is not None:
                    return cached
                inst = super().__new__(cls)
                cls._singleton = inst
                return inst
        return super().__new__(cls)

    def __init__(self, **attrs: Any) -> None:
        param_defs = _signature(type(self))
        attr_defs = {pd.name: pd for pd in param_defs if pd.kind == "attribute"}
        for k, v in attrs.items():
            if k not in attr_defs:
                raise TypeError(f"{type(self).__name__}: unknown attribute {k!r}")
            setattr(self, k, _normalize_attr(k, v))

        missing = set(attr_defs) - set(attrs)
        for m in list(missing):
            pd = attr_defs[m]
            if pd.has_default:
                setattr(self, m, _normalize_attr(m, pd.default))
                missing.discard(m)
        if missing:
            raise TypeError(f"{type(self).__name__}: missing attribute(s) {sorted(missing)}")

    def __repr__(self) -> str:
        infos = type(self).params()
        attr_bits = [
            f"{p.name}={getattr(self, p.name, '?')!r}" for p in infos if p.kind == "attribute"
        ]
        return f"{type(self).__name__}({', '.join(attr_bits)})"

    @classmethod
    def params(cls) -> list[ParameterInfo]:
        """``ParameterInfo`` projection of ``cls``'s ``ParamDef`` signature.

        Uses the schema signature attached by ``@register_op`` when
        present (`_signature`); otherwise reflects ``ParamDef``
        class-body descriptors directly (base→derived MRO order,
        derived redeclaration overrides in place).
        """
        cached = Op._params_cache.get(cls)
        if cached is not None:
            return cached
        infos = [
            ParameterInfo(name=pd.name, kind=pd.kind, type=pd.annotation) for pd in _signature(cls)
        ]
        Op._params_cache[cls] = infos
        return infos


__all__ = [
    "Op",
    "OpCapability",
    "ParameterInfo",
    "capabilities_of",
    "op_identifier",
    "supported_op_capabilities",
]
