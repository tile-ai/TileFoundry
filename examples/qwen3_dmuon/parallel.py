"""HSDP owner reduction and TP update publication on a two-by-two-by-two mesh."""

from tilefoundry.ir import types
from tilefoundry.ir.hir.sharding import Collective

MESH = types.Mesh((types.Topology("gpu", 8),), types.Layout((2, 2, 2)), ("replica", "shard", "tp"))


def gradient_operations():
    """Reduce four data contributions per TP shard and gather at the TP owner."""
    for replica in range(2):
        for tp in range(2):
            yield Collective(participants=MESH[replica, :, tp], context=f"shard-{replica}-{tp}",
                             kind="reduce", reduction="avg", root=0)
    for tp in range(2):
        yield Collective(participants=MESH[:, 0, tp], context=f"replica-{tp}",
                         kind="reduce", reduction="avg", root=0)
    yield Collective(participants=MESH[0, 0, :], context="tp-owner", kind="gather", root=0, axis=0)


def publication_operations():
    """Scatter the updated matrix, then publish through replica and shard groups."""
    yield Collective(participants=MESH[0, 0, :], context="tp-owner", kind="scatter", root=0, axis=0)
    for tp in range(2):
        yield Collective(participants=MESH[:, 0, tp], context=f"replica-{tp}", kind="broadcast", root=0)
    for replica in range(2):
        for tp in range(2):
            yield Collective(participants=MESH[replica, :, tp], context=f"shard-{replica}-{tp}", kind="broadcast", root=0)


def topology():
    """Describe logical nodes separately from the actual single-host placement."""
    return {
        "physical_hosts": 1,
        "logical_nodes": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "rank_coordinates": [dict(rank=i, replica=i // 4, shard=(i // 2) % 2, tp=i % 2) for i in range(8)],
        "scope": "HSDP and TP at the optimizer boundary; forward/backward reference remains replicated",
    }
