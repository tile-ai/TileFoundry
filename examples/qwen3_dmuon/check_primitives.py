"""Compare every supported tensor primitive against real NCCL on eight ranks."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

from tilefoundry import core
from tilefoundry.interfaces import evaluator
from tilefoundry.ir import types
from tilefoundry.ir.hir import sharding

from model import collective_function
from runtime import Runtime


def cases(mesh):
    """Exercise dtype, root, rank order, ragged traffic and empty payload contracts."""
    results = []
    size = len(types.Placement.from_mesh(mesh))
    rooted = {"broadcast", "reduce", "gather", "scatter"}
    reducing = {"reduce", "all_reduce", "reduce_scatter"}
    kinds = ("broadcast", "reduce", "all_reduce", "gather", "scatter", "all_gather", "reduce_scatter")
    patterns = dict(
        all_to_all=tuple((1,) * size for _ in range(size)),
        ragged=tuple(tuple((i * j + i) % 3 if i % 3 else 0 for j in range(size)) for i in range(size)),
        ring=tuple(tuple(i + 1 if j == (i + 1) % size else 0 for j in range(size)) for i in range(size)),
        p2p=tuple(tuple(4 if (i, j) == (0, size - 1) else 0 for j in range(size)) for i in range(size)),
        self=tuple(tuple(3 if (i, j) == (2, 2) else 0 for j in range(size)) for i in range(size)),
        zero=tuple((0,) * size for _ in range(size)),
    )
    for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64):
        for kind in kinds:
            reductions = (("sum", "avg", "max") if dtype.is_floating_point else ("sum", "max"))
            for reduction in reductions if kind in reducing else (None,):
                op = sharding.Collective(participants=mesh, context="ordered", kind=kind,
                                         root=size - 2 if kind in rooted else None, reduction=reduction)
                results.append((op, (size,), dtype))
        for counts in patterns.values():
            op = sharding.Exchange(participants=mesh, context="ordered", counts=counts)
            results.append((op, (max(map(sum, counts)),), dtype))
        op = sharding.Collective(participants=mesh, context="scalar", kind="all_reduce", reduction="sum")
        results.append((op, (), dtype))
    for kind in kinds:
        op = sharding.Collective(participants=mesh, context="empty", kind=kind,
                                 root=size - 2 if kind in rooted else None,
                                 reduction="sum" if kind in reducing else None)
        results.append((op, (0,), torch.float32))
    for kind in ("gather", "scatter", "all_gather", "reduce_scatter"):
        op = sharding.Collective(participants=mesh, context="axis", kind=kind, axis=-1,
                                 root=size - 2 if kind in rooted else None,
                                 reduction="sum" if kind in reducing else None)
        results.append((op, (2, size), torch.float32))
    op = sharding.Collective(participants=mesh[:, 0], context="subgroup", kind="broadcast", root=1)
    results.append((op, (3,), torch.int64))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", 0)) != 8:
        parser.error("launch with torchrun --nproc-per-node=8")
    rank = int(os.environ["RANK"])
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"rank{rank}.json"
    if destination.exists():
        raise FileExistsError("select a fresh output directory")
    os.environ.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,NET,P2P,ENV",
                      NCCL_DEBUG_FILE=str(args.output.resolve() / f"nccl-rank{rank}.log"))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    mesh = types.Mesh((types.Topology("gpu", 8),), types.Layout((2, 4), (4, 1)), ("a", "b"))
    members = types.Placement.from_mesh(mesh)
    tests = cases(mesh)
    runtime = Runtime([op for op, _, _ in tests])
    reports = []
    for op, shape, dtype in tests:
        places = types.Placement.from_mesh(op.participants)
        ranks = tuple(p.ids["gpu"] for p in places)
        pieces = tuple((p, torch.arange(math.prod(shape)).reshape(shape).to(dtype) + p.ids["gpu"])
                       for p in members)
        function = collective_function(op, shape, dtype)
        expected = evaluator.evaluate(function, core.ShardTensor(function.params[0].type, pieces))
        placement = types.Placement({"gpu": rank})
        source = dict((p.ids["gpu"], tensor) for p, tensor in pieces)[rank]
        local = source.to(device)
        if rank in ranks:
            if isinstance(op, sharding.Exchange) and not sum(op.counts[ranks.index(rank)]):
                local = None
            elif isinstance(op, sharding.Collective) and op.kind in {"broadcast", "scatter"} and rank != ranks[op.root]:
                local = None
        actual, result_shape = runtime.apply(op, local, shape, device, dtype=dtype)
        if rank not in ranks:
            torch.testing.assert_close(actual.cpu(), source, rtol=0, atol=0)
        elif placement in expected.placements:
            torch.testing.assert_close(actual.cpu(), expected.to_local(placement), rtol=0, atol=0)
            assert tuple(actual.shape) == result_shape and actual.dtype == dtype
        else:
            assert actual is None
        reports.append(dict(kind="exchange" if isinstance(op, sharding.Exchange) else op.kind,
                            dtype=str(dtype), input_shape=shape, output_shape=result_shape,
                            context=op.context))
    result = dict(status="passed", rank=rank, cases=len(tests), checks=reports,
                  ordered_ranks=[p.ids["gpu"] for p in members], torch=torch.__version__,
                  nccl=torch.cuda.nccl.version(), transport=os.environ.get("NCCL_NET", "native"),
                  trace=runtime.trace)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"Passed {len(tests)} primitive cases on all eight ranks")


if __name__ == "__main__":
    main()
