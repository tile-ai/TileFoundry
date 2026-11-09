"""CPU (host) target codegen: the host-entry linkable module, and how it is called."""

from __future__ import annotations

from . import abi, launch, module
from .context import Context
