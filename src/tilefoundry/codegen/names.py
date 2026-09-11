"""The identifiers codegen invents, spelled in one place.

A generated name has to stay clear of two other name sets at once: the ones a
user writes in a kernel, and the ones C and C++ reserve for the implementation
(anything with a leading double underscore). One project prefix on each name we
generate clears both, and the user's own parameters keep their names.
"""

from __future__ import annotations

PREFIX = "tilefoundry_"


def _identifier(name: str) -> str:
    """*name* as a plain C identifier -- a mangled variant's ``$`` is not one."""
    return name.replace("$", "__")


def host_entry(name: str) -> str:
    """The C++ symbol of *name*'s host entry, which ``main`` would collide with."""
    return f"{PREFIX}{_identifier(name)}_host"


def launch_shim(name: str) -> str:
    """The ``extern "C"`` symbol of the shim that launches kernel *name*."""
    return f"{PREFIX}{_identifier(name)}_launch"


def device_kernel(name: str) -> str:
    """The ``__global__`` symbol of the kernel compiled from *name*."""
    return f"{PREFIX}{_identifier(name)}_kernel"


def placed_id_param() -> str:
    """The parameter carrying the card's id, which no card can read for itself."""
    return f"{PREFIX}gpu_program_id"


def meta_param() -> str:
    """The parameter carrying the ids a placed kernel hands its block."""
    return f"{PREFIX}meta"


__all__ = ["PREFIX", "device_kernel", "host_entry", "launch_shim", "meta_param", "placed_id_param"]
