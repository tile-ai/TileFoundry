"""``@func(topologies=(...))`` + ``with Mesh(topology="...", ...)`` parser tests.

A declaration list names the levels a body may open a Mesh on, and a body's
string topology is resolved against it. The positive path is exercised by every
``with Mesh(topology=...)`` scene in the shard-sugar tests and by the corpus
print / re-import trip; what only a source-level test can pin is the failure of
a declaration list that cannot answer a body's request.
"""

from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F403
from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.types.shard import Layout, Mesh, Topology


def test_topology_errors() -> None:
    """Duplicate topology name + unknown Mesh topology both raise."""
    with pytest.raises(VerifyError, match="duplicate topology name"):

        @func(topologies=(Topology("cta", 128), Topology("cta", 64)))
        def _dup(a: Tensor[(1, 1536), "f32"]) -> Tensor[(1, 1536), "f32"]:
            return a

    with pytest.raises(VerifyError, match="topology.*not declared"):

        @func(topologies=(Topology("cta", 128),))
        def _unk(a: Tensor[(1, 1536), "f32"]) -> Tensor[(1, 1536), "f32"]:
            with Mesh(
                topology="nonexistent",
                layout=Layout(shape=(128,), strides=(1,)),
            ) as m:  # noqa: F841
                return a
