"""Load the packages that register implementations, once, when ``tilefoundry`` is imported.

Importing a package runs the registrations its modules declare, and ``target.load``
loads the installed Target packages. A key with no registration raises
``MissingHandlerError`` where it is looked up.
"""

from __future__ import annotations

from . import codegen, inspection, parser, passes, runtime, target
from .ir import hir, tir

registering_packages = (codegen, hir, inspection, parser, passes, runtime, tir)

target.load()

__all__ = ["registering_packages"]
