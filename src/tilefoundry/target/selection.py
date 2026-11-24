"""The selection of one Target's architecture and device from its hardware documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tilefoundry import dispatch

from .hardware import IncompatiblePairError, ResolvedResource, UnknownDocumentError
from .registry import catalog
from .spec import HardwareSpec
from .target import Device


def select(
    value: Any,
    base_type: type,
    *,
    role: str,
    hardware: HardwareSpec,
) -> ResolvedResource:
    """Resolve an ID or path through one Target's spec, or retain a direct value."""
    if isinstance(value, str):
        try:
            resolved = hardware.resolve(value)
        except UnknownDocumentError:
            for target_type in catalog.registered().values():
                other = target_type.hardware
                if other is None or other is hardware:
                    continue
                if value in other.documents():
                    raise UnknownDocumentError(
                        f"{role} {value!r} is a hardware document owned by "
                        f"{dispatch.spelled(target_type)}, not {role.partition('.')[0]}"
                    ) from None
            raise
    elif isinstance(value, Path):
        resolved = hardware.read(value)
    elif isinstance(value, base_type):
        return ResolvedResource(value=value)
    else:
        article = "an" if base_type.__name__[0].lower() in "aeiou" else "a"
        raise TypeError(
            f"{role} must be an installed ID string, a document path, or "
            f"{article} {base_type.__name__}; got {type(value).__name__}"
        )
    if not isinstance(resolved.value, base_type):
        raise UnknownDocumentError(
            f"{role} got hardware document {resolved.id!r}, which builds "
            f"{type(resolved.value).__name__}; expected {base_type.__name__}"
        )
    return resolved


def available_device_ids(hardware: HardwareSpec) -> tuple[str, ...]:
    """Return devices whose sole compatible architecture is available."""
    documents = hardware.documents()
    return tuple(
        document.id
        for document in sorted(documents.values(), key=lambda item: item.id)
        if document.kind == "device"
        and len(document.compatibility) == 1
        and document.compatibility[0] in documents
        and documents[document.compatibility[0]].kind == "architecture"
    )


def architecture_of(
    device: Any,
    *,
    device_type: type[Device],
    role: str,
    hardware: HardwareSpec,
) -> str:
    """Return the sole architecture declared by one device document."""
    target_name = role.partition(".")[0]
    if isinstance(device, device_type):
        raise ValueError(
            f"{target_name}: a {device_type.__name__} supplied directly carries "
            "no document to read a compatible architecture from; name the "
            "architecture as well"
        )
    architectures = select(
        device, device_type, role=role, hardware=hardware
    ).document.compatibility
    if len(architectures) != 1:
        raise IncompatiblePairError(
            f"device {device!r} declares {list(architectures)} as compatible "
            "architectures; name the one to build against"
        )
    return architectures[0]


def check_compatible(
    architecture: ResolvedResource, device: ResolvedResource
) -> None:
    """Require that an architecture and device declare each other usable."""
    if architecture.document is None or device.document is None:
        raise IncompatiblePairError("compatibility requires two hardware documents")
    if architecture.document.kind != "architecture":
        raise IncompatiblePairError(
            f"{architecture.id}: expected an architecture document, "
            f"got kind {architecture.document.kind!r}"
        )
    if device.document.kind != "device":
        raise IncompatiblePairError(
            f"{device.id}: expected a device document, got kind {device.document.kind!r}"
        )
    device_allows = device.document.compatibility
    architecture_allows = architecture.document.compatibility
    if device_allows and architecture.id not in device_allows:
        raise IncompatiblePairError(
            f"device {device.id!r} declares compatibility with "
            f"{list(device_allows)}, not {architecture.id!r}"
        )
    if architecture_allows and device.id not in architecture_allows:
        raise IncompatiblePairError(
            f"architecture {architecture.id!r} declares compatibility with "
            f"{list(architecture_allows)}, not {device.id!r}"
        )
    if not device_allows and not architecture_allows:
        raise IncompatiblePairError(
            f"neither {architecture.id!r} nor {device.id!r} declares the pair "
            "compatible; one side must name the other"
        )


__all__ = ["architecture_of", "available_device_ids", "check_compatible", "select"]
