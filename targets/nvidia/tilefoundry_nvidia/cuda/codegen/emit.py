"""CUDA emitter handler autodiscovery.

Importing this module loads per-Op emitters under ``codegen/tir/`` and ``codegen/hir/``
before codegen runs. What a function is called and what it takes are answered in
``codegen.signature`` and ``tilefoundry_nvidia.cuda.codegen.abi``, not here.
"""

from __future__ import annotations

import importlib
import os
import pkgutil

_tir_path = os.path.dirname(__file__)


def _discover(subdir: str, prefix: str) -> None:
    full = os.path.join(_tir_path, subdir)
    if not os.path.isdir(full):
        return
    for _finder, _name, _ispkg in pkgutil.iter_modules([full], prefix=prefix):
        importlib.import_module(_name)


_discover("tir/stmts", f"{__package__}.tir.stmts.")
_discover("tir/memory", f"{__package__}.tir.memory.")
_discover("tir/nn", f"{__package__}.tir.nn.")
_discover("tir", f"{__package__}.tir.")

_discover("hir", f"{__package__}.hir.")
