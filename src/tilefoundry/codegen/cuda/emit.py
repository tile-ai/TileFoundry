"""CUDA emitter handler autodiscovery.

Importing this module loads every registered per-Op emitter under ``cuda/tir/``
so its handler is registered against this target before codegen runs. What a
function is called and what it takes are answered in ``codegen.signature`` and
``codegen.cuda.abi``, not here.
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil

_log = logging.getLogger(__name__)
_tir_path = os.path.dirname(__file__)


def _discover(subdir: str, prefix: str) -> None:
    full = os.path.join(_tir_path, subdir)
    if not os.path.isdir(full):
        return
    for _finder, _name, _ispkg in pkgutil.iter_modules([full], prefix=prefix):
        try:
            importlib.import_module(_name)
        except Exception:
            _log.debug("codegen autodiscovery: skip %s", _name, exc_info=True)


_discover("tir/stmts", "tilefoundry.codegen.cuda.tir.stmts.")
_discover("tir/memory", "tilefoundry.codegen.cuda.tir.memory.")
_discover("tir/nn", "tilefoundry.codegen.cuda.tir.nn.")
_discover("tir", "tilefoundry.codegen.cuda.tir.")
