"""NCCL realization of the example's authored communication primitives."""

import torch
import torch.distributed as dist

from tilefoundry.ir import types
from tilefoundry.ir.hir import sharding

from model import collective_function


class Runtime:
    """Bind isolated contexts once, preserving authored participant order."""

    def __init__(self, operations):
        self.groups = {}
        self.trace = []
        for op in operations:
            ranks = tuple(p.ids["gpu"] for p in types.Placement.from_mesh(op.participants))
            if op.context in self.groups:
                if self.groups[op.context][0] != ranks:
                    raise ValueError("a communicator context cannot change participants")
            else:
                self.groups[op.context] = ranks, dist.new_group(list(ranks), sort_ranks=False)

    def apply(self, op, value, shape, device, *, dtype=torch.float32):
        """Return a valid local output or None when only another root receives it."""
        ranks, group = self.groups[op.context]
        rank = dist.get_rank()
        shape = tuple(collective_function(op, shape, dtype).return_type.shape)
        if rank not in ranks:
            return value, shape
        if value is not None and value.dtype != dtype:
            raise TypeError(f"payload dtype {value.dtype} differs from declared {dtype}")
        local = ranks.index(rank)
        if isinstance(op, sharding.Exchange):
            output = torch.zeros(shape, dtype=dtype, device=device)
            send_offset = receive_offset = 0
            operations = []
            for peer, peer_rank in enumerate(ranks):
                sent, received = op.counts[local][peer], op.counts[peer][local]
                if peer == local:
                    if sent:
                        output[receive_offset:receive_offset + received].copy_(value[send_offset:send_offset + sent])
                else:
                    if sent:
                        operations.append(dist.P2POp(dist.isend, value.narrow(0, send_offset, sent), peer_rank, group))
                    if received:
                        operations.append(dist.P2POp(dist.irecv, output.narrow(0, receive_offset, received), peer_rank, group))
                send_offset += sent
                receive_offset += received
            for request in dist.batch_isend_irecv(operations) if operations else ():
                request.wait()
            self.trace.append(dict(context=op.context, kind="exchange", participants=ranks,
                                   communicator_rank=local, counts=op.counts, valid_count=receive_offset,
                                   dtype=str(dtype)))
            return output, shape
        root = ranks[op.root] if op.root is not None else None
        if op.kind == "reduce":
            output = value.clone()
            dist.reduce(output, dst=root, group=group, op={"avg": dist.ReduceOp.AVG, "sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX}[op.reduction])
            output = output if rank == root else None
        elif op.kind == "gather":
            buffers = [torch.empty_like(value) for _ in ranks] if rank == root else None
            dist.gather(value, gather_list=buffers, dst=root, group=group)
            output = torch.cat(buffers, dim=op.axis) if buffers is not None else None
        elif op.kind == "scatter":
            output = torch.empty(shape, dtype=dtype, device=device)
            buffers = [x.contiguous() for x in value.chunk(len(ranks), dim=op.axis)] if rank == root else None
            dist.scatter(output, scatter_list=buffers, src=root, group=group)
        elif op.kind == "broadcast":
            output = value.clone() if rank == root else torch.empty(shape, dtype=dtype, device=device)
            dist.broadcast(output, src=root, group=group)
        elif op.kind == "all_reduce":
            output = value.clone()
            dist.all_reduce(output, group=group, op={"avg": dist.ReduceOp.AVG, "sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX}[op.reduction])
        elif op.kind == "all_gather":
            buffers = [torch.empty_like(value) for _ in ranks]
            dist.all_gather(buffers, value, group=group)
            output = torch.cat(buffers, dim=op.axis)
        elif op.kind == "reduce_scatter":
            output = torch.empty(shape, dtype=dtype, device=device)
            dist.reduce_scatter(output, [x.contiguous() for x in value.chunk(len(ranks), dim=op.axis)],
                                group=group, op={"avg": dist.ReduceOp.AVG, "sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX}[op.reduction])
        else:
            raise ValueError(f"unsupported example realization {op.kind}")
        self.trace.append(dict(context=op.context, kind=op.kind, participants=ranks, root=root,
                               communicator_rank=local, output_valid=output is not None, dtype=str(dtype)))
        return output, shape
