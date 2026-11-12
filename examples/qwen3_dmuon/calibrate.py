"""Measure communication service times for example-only virtual-grid experiments."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("native", "socket"), default="native")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 2 or args.warmup < 1:
        parser.error("require iterations >= 2 and warmup >= 1")
    if args.transport == "socket":
        os.environ.update(NCCL_NET="Socket", NCCL_P2P_DISABLE="1", NCCL_SHM_DISABLE="1",
                          NCCL_NVLS_ENABLE="0", NCCL_MNNVL_ENABLE="0")
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=300))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise ValueError("this calibration requires the validated eight-GPU node")
    if args.output.exists():
        raise FileExistsError(args.output)
    records = []
    for size in (2, 4, 8):
        group = None
        for start in range(0, world, size):
            ranks = list(range(start, start + size))
            candidate = dist.new_group(ranks)
            if rank in ranks:
                group = candidate
        peers = dist.get_process_group_ranks(group)
        local = dist.get_rank(group)
        for nbytes in (4096, 1048576, 16777216, 67108864):
            count = nbytes // 4
            source = torch.ones(count, device=device)
            gathered = torch.empty(count * size, device=device)
            target = torch.empty_like(source)
            for kind in ("all_gather", "reduce_scatter", "all_reduce", "broadcast", "reduce", "exchange", "p2p"):
                def operation():
                    if kind == "all_gather":
                        dist.all_gather_into_tensor(gathered, source, group=group)
                    elif kind == "reduce_scatter":
                        dist.reduce_scatter_tensor(target, gathered, group=group)
                    elif kind == "all_reduce":
                        dist.all_reduce(source, group=group)
                    elif kind == "broadcast":
                        dist.broadcast(source, src=peers[0], group=group)
                    elif kind == "reduce":
                        dist.reduce(source, dst=peers[0], group=group)
                    elif kind == "exchange":
                        dist.all_to_all_single(target, source, group=group)
                    else:
                        requests = dist.batch_isend_irecv([
                            dist.P2POp(dist.isend, source, peers[(local + 1) % size], group),
                            dist.P2POp(dist.irecv, target, peers[(local - 1) % size], group)])
                        for request in requests:
                            request.wait()

                for _ in range(args.warmup):
                    operation()
                dist.barrier()
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(args.iterations):
                    operation()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - started) / args.iterations
                times = [None] * world
                dist.all_gather_object(times, elapsed)
                records.append(dict(kind=kind, size=size, bytes=nbytes, seconds=max(times), rank_seconds=times))
    hosts = [None] * world
    dist.all_gather_object(hosts, os.uname().nodename)
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(transport=args.transport, iterations=args.iterations,
                                               warmup=args.warmup, physical_hosts=len(set(hosts)),
                                               device=torch.cuda.get_device_name(), torch=torch.__version__,
                                               nccl=torch.cuda.nccl.version(), records=records,
                                               scope="concurrent disjoint groups; max rank wall time per call; FP32",
                                               bytes="local input except reduce_scatter uses local output bytes",
                                               emulated=args.transport == "socket"), indent=2) + "\n")
        print(json.dumps(dict(output=str(args.output), cases=len(records))))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
