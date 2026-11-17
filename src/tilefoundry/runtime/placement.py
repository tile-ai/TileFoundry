"""The placement of the process a launcher started."""

from __future__ import annotations

import os

from tilefoundry.ir import types

__all__ = ["placement_from_env"]


def placement_from_env(topology: str = "gpu") -> types.Placement:
    """The peer this process is, out of the ones a launcher started.

    ``torchrun``, ``deepspeed`` and ``mpirun`` all put the unit id of this process in the
    environment as ``LOCAL_RANK`` before the script runs, which is the one moment the
    answer is settled.
    """
    unit_id = os.environ.get("LOCAL_RANK")
    if unit_id is None:
        raise ValueError(
            "placement_from_env: no LOCAL_RANK in the environment; a launcher "
            "sets it, so pass Placement({topology!r}: unit_id) yourself instead"
        )
    return types.Placement({topology: int(unit_id)})
