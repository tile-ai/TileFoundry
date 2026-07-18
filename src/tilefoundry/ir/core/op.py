"""Op base class — every callable IR Op inherits from ``Op`` and declares
its parameters as ``ParamDef`` class-body descriptors.

Post-M1c the legacy ``Param[T, "kind"]`` annotation form, the
metaclass auto-register, and the textual annotation parser have been
removed. ``Op.params()`` reflects ``ParamDef`` descriptors
in MRO order; ``Op.__init__`` accepts attribute kwargs only and honors
``ParamDef.default`` for omitted attributes.

Registration is opt-in via ``@register_op`` (see
``tilefoundry.ir.core.register``). The metaclass-driven flat ``name`` /
``category`` indices have been folded into the OpSchema list-per-name
registry — ``resolve_op`` / ``resolve_stmt`` derive their lookups from
``_schemas_by_dialect_name`` directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from tilefoundry.ir.core.param_def import MISSING, ParamDef
from tilefoundry.ir.types.storage import resolve_storage

_ParamKind = Literal["input", "attribute"]


def _normalize_attr(name: str, value: Any) -> Any:
    """Normalise known typed attributes at the IR construction boundary.

    A ``storage`` attribute is coerced from a legacy string alias to
    ``StorageKind | None`` so an Op instance never carries a raw string.
    """
    if name == "storage":
        return resolve_storage(value)
    return value


@dataclass(frozen=True)
class ParameterInfo:
    """Lightweight reflection record for a declared Op parameter."""
    name: str
    kind: _ParamKind
    type: Any


class Op:
    """All Op classes inherit from this. Reflection-based param discovery."""

    _params_cache: ClassVar[dict[type, list[ParameterInfo]]] = {}

    def __new__(cls, **attrs: Any):
        # Singleton cache for no-attribute Ops (spec 001). Multiple
        # ``Foo()`` calls with no kwargs return the same instance so
        # downstream code can use ``is`` for op identity. Replaces the
        # pre-M1c metaclass ``__call__`` singleton path.
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
        infos = type(self).params()
        attr_infos = {p.name: p for p in infos if p.kind == "attribute"}
        for k, v in attrs.items():
            if k not in attr_infos:
                raise TypeError(f"{type(self).__name__}: unknown attribute {k!r}")
            object.__setattr__(self, k, _normalize_attr(k, v))
        # Apply class-level defaults for missing attribute params.
        missing = set(attr_infos) - set(attrs)
        for m in list(missing):
            cls_val = None
            present = False
            for klass in type(self).__mro__:
                if m in klass.__dict__:
                    cls_val = klass.__dict__[m]
                    present = True
                    break
            if not present:
                continue
            if isinstance(cls_val, ParamDef):
                if cls_val.default is not MISSING:
                    object.__setattr__(self, m, _normalize_attr(m, cls_val.default))
                    missing.discard(m)
        if missing:
            raise TypeError(
                f"{type(self).__name__}: missing attribute(s) {sorted(missing)}"
            )

    def __repr__(self) -> str:
        infos = type(self).params()
        attr_bits = [
            f"{p.name}={getattr(self, p.name, '?')!r}"
            for p in infos
            if p.kind == "attribute"
        ]
        return f"{type(self).__name__}({', '.join(attr_bits)})"

    @classmethod
    def params(cls) -> list[ParameterInfo]:
        """Reflect ``ParamDef`` class-body descriptors into ``ParameterInfo``.

        Walks MRO base→derived so subclass fields list **after** base
        fields in declaration order. Same-name fields declared at a
        more-derived class **override** the base entry in place
        (preserving the base position in the resulting tuple) — this
        matches the "derived override wins" contract used by all
        downstream callers.
        """
        cached = Op._params_cache.get(cls)
        if cached is not None:
            return cached
        infos: list[ParameterInfo] = []
        index: dict[str, int] = {}
        for klass in reversed(cls.__mro__):
            for attr_name, value in klass.__dict__.items():
                if attr_name.startswith("_"):
                    continue
                if not isinstance(value, ParamDef):
                    continue
                pi = ParameterInfo(
                    name=attr_name, kind=value.kind, type=value.annotation
                )
                if attr_name in index:
                    # Derived class re-declared this field → override
                    # the base entry at its existing position.
                    infos[index[attr_name]] = pi
                else:
                    index[attr_name] = len(infos)
                    infos.append(pi)
        Op._params_cache[cls] = infos
        return infos


__all__ = ["Op", "ParameterInfo"]
