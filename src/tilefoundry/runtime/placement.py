"""The placement of the process a launcher started."""

from __future__ import annotations

import os

from tilefoundry.ir import types

__all__ = ["placement_from_env"]


def placement_from_env(topology: str = "gpu") -> types.Placement:
    """The peer this process is, out of the ones a launcher started.

    ``torchrun``, ``deepspeed`` and ``mpirun`` all put the local rank in
    the environment before the script runs, which is the one moment the
    answer is settled.
    """
    rank = os.environ.get("LOCAL_RANK")
    if rank is None:
        raise ValueError(
            "placement_from_env: no LOCAL_RANK in the environment; a launcher "
            "sets it, so pass Placement({topology!r}: rank) yourself instead"
        )
    return types.Placement({topology: int(rank)})
