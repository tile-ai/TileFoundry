"""Define callable IR operations from ``ParamDef`` descriptors.

``Op.params()`` reflects descriptors in MRO order; construction accepts only
attribute parameters. Registration is explicit through ``@register_op`` and
the ordered ``OpSchema`` registry.

See [core-ir §2.3](docs/spec/core-ir.md#23-op).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from tilefoundry.ir.core.register import get_registered_name
from tilefoundry.ir.core.param_def import ParamDef, _ParamKind, collect_param_defs
from tilefoundry.ir.types.storage import resolve_storage


def _signature(cls: type) -> tuple[ParamDef, ...]:
    """Signature.

    The ``ParamDef`` tuple for ``cls``: the attached ``OpSchema``'s
    ``signature`` when ``@register_op`` has run, else a fresh reflection
    walk (e.g. an unregistered test fixture subclassing ``Op`` directly).
    """
    schema = vars(cls).get("_op_schema")
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

    ``name=None`` imposes no named capability requirement; the target must still
    admit the instruction's execution topology. A carrier
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

æ 这个函数不对啊，不应该加这里
def _supports_execution_topology(target: object, op_type: type, capability: OpCapability) -> bool:
    """Require a real execution branch whose levels the target declares.

    Carrier variants must resolve through their bound declaration, just as
    instruction admission does; a carrier's optional mesh field is not its
    atom's execution contract.
    """
    from tilefoundry.ir.pattern import OrPattern  # noqa: PLC0415
    from tilefoundry.ir.pattern.utils import declared_execution_mesh  # noqa: PLC0415
    from tilefoundry.target.facts import TopologyFacts  # noqa: PLC0415

    get_facts = getattr(target, "get_facts", None)
    if get_facts is None:
        return False
    levels = {level.name for level in get_facts(TopologyFacts).topologies}
    if not levels:
        return False
    declaration = (
        op_type(**{capability.attribute: capability.declaration})
        if capability.attribute is not None
        else op_type
    )
    # This also expands wildcard memory levels from MemoryExecutionFacts.
    mesh = declared_execution_mesh(declaration, target)
    choices = mesh.patterns if isinstance(mesh, OrPattern) else (mesh,)
    return any(set(choice.topologies) <= levels for choice in choices)


def supported_op_capabilities(target: object) -> tuple[tuple[type, OpCapability], ...]:
    """Enumerate TIR variants admitted by target capabilities and execution levels."""
    from tilefoundry.ir.core.op_registry import iter_schemas  # noqa: PLC0415

    supported = target_capabilities(target)
    return tuple(
        (schema.op_class, capability)
        for schema in iter_schemas()
        if schema.dialect == "T" and schema.op_class is not None
        for capability in capabilities_of(schema.op_class)
        if (capability.name is None or capability.name in supported)
        and _supports_execution_topology(target, schema.op_class, capability)
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
    "supported_op_capabilities",
    "target_capabilities",
]
