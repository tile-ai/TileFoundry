"""DMuon materialization and updates over model-declared parameter domains."""

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from tilefoundry import utils

dmuon = utils.LazyModule("dmuon")


def state(model):
    """Materialize complete state after asynchronous publication finishes."""
    dmuon.wait_all_post_step_broadcasts(model)
    torch.cuda.synchronize()
    return dmuon.get_model_state_dict(model, cpu_offload=False, rank0_only=False)

def domain_view(members):
    """Keep executing modules and their names; only the namespace containers are new."""
    view = nn.Module()
    for path, module in members:
        parent = view
        *prefix, name = path.split(".")
        for part in prefix:
            if not hasattr(parent, part):
                parent.add_module(part, nn.Module())
            parent = getattr(parent, part)
        parent.add_module(name, module)
    return view


def configure(model, group, roles, *, overlap=False, replicate_group=None,
              shard_group=None, ranks=None, boundaries=None):
    """Dedicate attention matrices and individual expert projections within one stage."""
    if replicate_group is None:
        mesh = DeviceMesh.from_group(group, "cuda", mesh_dim_names=("shard",))
        shard_mesh, replicate_mesh = mesh, None
    else:
        mesh = DeviceMesh.from_group([replicate_group, shard_group], "cuda", mesh=ranks,
                                     mesh_dim_names=("replicate", "shard"))
        shard_mesh, replicate_mesh = mesh["shard"], mesh["replicate"]
    if boundaries is None:
        boundaries = set(model.model.layers.values())
        if model.first:
            boundaries.add(model.model.embed_tokens)
        if model.last:
            boundaries.update((model.model.norm, model.lm_head))

    def policy(name, parameter):
        role, dimensions = roles.get(name, ("adamw", None))
        if role == "expert":
            return dict(route="muon", matrix_layout=dmuon.MatrixLayout.view(*dimensions))
        if role == "attention":
            return dict(route="muon", muon_forward_unshard="broadcast")
        return dict(route="adamw")

    dmuon.dedicate_params(model, shard_mesh, replicate_mesh=replicate_mesh,
                         predicate=lambda name, p: p.requires_grad,
                         hook_boundary_predicate=lambda module: module in boundaries,
                         param_policy_fn=policy, process_group_policy="isolated")
    return dmuon.Muon(model, ns_backend="direct", replicate_async=overlap,
                      forward_prefetch_depth=1 if overlap else 0,
                      post_step_prefetch_groups=int(overlap and replicate_group is None),
                      first_step_progress_log=False)

class Optimizer:
    def __init__(self, stage, parallel, *, overlap):
        domains = stage.parameter_domains() if parallel.ep > 1 else ()
        self.models = tuple(domain_view(members) for members in domains) if domains else (stage,)
        boundaries = tuple({module for _, module in members} for members in domains) if domains else (None,)
        self.names = [tuple(dict(model.named_parameters())) for model in self.models]
        self.groups = (parallel.data, parallel.expert_group) if parallel.ep > 1 else (parallel.data,)
        self.parallel, self.dense_group = parallel, parallel.data
        count = stage.config.num_experts // parallel.ep
        self.expert_slice = slice(parallel.ep_rank * count, (parallel.ep_rank + 1) * count)
        roles = stage.parameter_roles()
        layouts = ((parallel.replicate, parallel.shard, parallel.ranks),
                   (parallel.expert_replicate, parallel.expert_shard, parallel.expert_ranks))
        self.optimizers = [configure(model, group, roles, overlap=overlap, boundaries=boundaries,
                                     replicate_group=replicate, shard_group=shard, ranks=ranks)
                           for model, group, boundaries, (replicate, shard, ranks)
                           in zip(self.models, self.groups, boundaries, layouts)]

    def zero_grad(self, **kwargs):
        for optimizer in self.optimizers:
            optimizer.zero_grad(**kwargs)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def snapshot(self):
        result = state(self.models[0])
        if len(self.models) == 1:
            return result
        for name, weight in state(self.models[1]).items():
            parts = [torch.empty_like(weight) for _ in range(self.parallel.ep)]
            dist.all_gather(parts, weight.contiguous(), group=self.parallel.ep_group)
            result[name] = torch.cat(parts, dim=0)
        return result
