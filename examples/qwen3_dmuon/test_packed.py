"""Independent packing and scheduling invariants for the distributed example."""

from types import SimpleNamespace

import pytest
import torch

from packed import Pack
from simulate import Graph, schedule_order


def test_document_boundaries_and_cross_cp_targets(monkeypatch):
    monkeypatch.setattr("packed.dist.all_reduce", lambda *args, **kwargs: None)
    parallel = SimpleNamespace(cp=2, cp_rank=0, data_rank=1, dense_size=4, data_size=2)
    pack = Pack(parallel, 128, 3, 0, "cpu", None)
    tokens = torch.arange(128)[None]
    targets = pack.targets(tokens)
    assert targets[0, 63] == 64
    assert targets[0, 79] == -100
    assert pack.positions[0, 80] == 0
    assert not pack.mask[0, 0, 80, :80].any()
    assert pack.mask[0, 0, 63, :64].all()
    assert not pack.mask[0, 0, 63, 64:].any()
    assert (targets != -100).sum() == pack.summary["valid_targets"]
    assert targets[0, -pack.summary["padding"]:].eq(-100).all()


def test_global_loss_weights_unequal_packs(monkeypatch):
    monkeypatch.setattr("packed.dist.all_reduce", lambda *args, **kwargs: None)
    config = SimpleNamespace(num_experts=4, num_experts_per_tok=2, router_aux_loss_coef=0.001)
    packs, logits, routers, targets = [], [], [], []
    torch.manual_seed(17)
    for data in range(2):
        parallel = SimpleNamespace(cp=2, cp_rank=0, data_rank=data, dense_size=4, data_size=2)
        pack = Pack(parallel, 64, 1, 2, "cpu", None)
        packs.append(pack)
        logits.append(torch.randn(1, 64, 8))
        routers.append(torch.randn(1, 2, 64, 4).softmax(-1))
        targets.append(pack.targets(torch.randint(8, (1, 64))))
    denominator = sum(pack.valid_targets for pack in packs)
    assert packs[0].valid_targets != packs[1].valid_targets
    reference, actual = [], []
    for pack, logit, router, target in zip(packs, logits, routers, targets):
        pack.denominator = denominator
        reference.append(pack.loss((logit, router), target, config, reference=True))
        pack.parallel.cp_group = None
        monkeypatch.setattr("packed.gather", lambda *args, router=router, **kwargs: router)
        for cp_rank in range(2):
            pack.start = cp_rank * pack.length
            actual.append(pack.loss((pack.local(logit), pack.local(router, 2)), pack.local(target), config))
    torch.testing.assert_close(torch.stack(actual).mean(), torch.stack(reference).mean())


def test_schedule_dependencies_and_shared_fabric():
    graph = Graph()
    graph.add("a", 3, resources=("fabric/0",), name="send", stage=0, lane=1, step=0)
    graph.add("b", 2, resources=("fabric/0",), name="send", stage=1, lane=1, step=0)
    graph.add("c", 4, deps=("a",), resources=("gpu/0",), name="compute", stage=0, lane=0, step=0)
    elapsed, events = graph.run()
    assert elapsed == 7
    sends = sorted((e["ts"], e["ts"] + e["dur"]) for e in events if e["name"] == "send")
    assert sends[0][1] <= sends[1][0]
    graph.add("cycle", 1, deps=("cycle",), name="invalid", stage=0, lane=0, step=0)
    with pytest.raises(ValueError, match="dependency cycle"):
        graph.run()


@pytest.mark.parametrize("stages,microbatches", [(2, 4), (4, 16), (8, 32)])
def test_reused_schedule_has_all_forward_backward_pairs(stages, microbatches):
    order = schedule_order(stages, microbatches)
    for actions in order.values():
        for kind in ("F", "B"):
            assert [m for k, m in actions if k == kind] == list(range(microbatches))
        completed = set()
        for kind, microbatch in actions:
            if kind == "B":
                assert microbatch in completed
            else:
                completed.add(microbatch)
