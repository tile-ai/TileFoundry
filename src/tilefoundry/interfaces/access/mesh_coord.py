"""Access relations of ``MeshCoord``: it touches no tensor."""

from __future__ import annotations

from tilefoundry.ir import dim

from .builders import readnone_relations
from .registry import register_relation

__all__: list[str] = []

register_relation(dim.MeshCoord)(readnone_relations(lambda types: ()))
