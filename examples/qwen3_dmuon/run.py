"""Run Qwen training gradients through modeled or NCCL dmuon communication."""

import argparse
import json
import os
from pathlib import Path

from config import FULL, SOURCES, TINY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("model", "nccl", "shapes"), default="model")
    parser.add_argument("--transport", choices=("native", "socket"), default="native")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--check-dmuon", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(".tmp/qwen3-dmuon/example"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.mode == "shapes":
        print(json.dumps(dict(parameters=FULL.parameter_count(), q_width=FULL.q_heads * FULL.head_dim,
                              kv_width=FULL.kv_heads * FULL.head_dim, sources=SOURCES), indent=2))
        return
    if args.transport == "socket":
        os.environ.update(NCCL_NET="Socket", NCCL_P2P_DISABLE="1", NCCL_SHM_DISABLE="1",
                          NCCL_NVLS_ENABLE="0", NCCL_MNNVL_ENABLE="0")
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", 0))
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,GRAPH,NET,P2P,COLL,ENV"
    os.environ["NCCL_DEBUG_FILE"] = str(args.output.resolve() / f"nccl-rank{rank}.log")
    os.environ["NCCL_TOPO_DUMP_FILE"] = str(args.output.resolve() / f"topology-rank{rank}.xml")
    import torch
    import transformers
    from model import apply_collective, reference_model, training_gradient
    from optimizer import orthogonalize, step
    from parallel import gradient_operations, publication_operations, topology

    torch.set_num_threads(1)
    gradients, publication = tuple(gradient_operations()), tuple(publication_operations())
    losses, errors, gradient_errors, momentum = [], [], [], None
    runtime = None
    if args.mode == "nccl":
        from datetime import timedelta
        import torch.distributed as dist
        from runtime import Runtime

        if int(os.environ.get("WORLD_SIZE", 0)) != 8:
            parser.error("NCCL mode requires torchrun --nproc-per-node=8")
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=90))
        runtime = Runtime((*gradients, *publication))
    else:
        device = "cpu"
    model = reference_model(device)
    weight = model.model.layers[0].self_attn.q_proj.weight
    for index in range(args.steps):
        if runtime is None:
            local = [training_gradient(model, data_rank, index) for data_rank in range(4)]
            expected_gradient = torch.stack([g for _, g in local]).mean(0)
            payloads = {r: local[r // 2][1].chunk(2, dim=0)[r % 2].clone() for r in range(8)}
            for op in gradients:
                payloads = apply_collective(op, payloads)
            gathered = payloads[0]
            torch.testing.assert_close(gathered, expected_gradient, rtol=1e-5, atol=1e-7)
            gradient_errors.append(float((gathered - expected_gradient).abs().max()))
            updated, momentum = step(weight.detach(), gathered, momentum)
            if args.check_dmuon:
                from dmuon.optim.newton_schulz import direct_newton_schulz

                ns_input = gathered + 0.95 * momentum
                torch.testing.assert_close(orthogonalize(ns_input), direct_newton_schulz(ns_input), rtol=0, atol=0)
            expected = updated.clone()
            payloads = {0: updated}
            for op in publication:
                payloads = apply_collective(op, payloads)
            for r, piece in payloads.items():
                torch.testing.assert_close(piece, expected.chunk(2, dim=0)[r % 2], rtol=0, atol=0)
            errors.append(float((updated - expected).abs().max()))
            losses.append([float(loss) for loss, _ in local])
            full = torch.cat((payloads[0], payloads[1]), dim=0)
        else:
            loss, gradient = training_gradient(model, rank // 2, index)
            expected_gradient = gradient.clone()
            dist.all_reduce(expected_gradient, op=dist.ReduceOp.AVG)
            value = gradient.chunk(2, dim=0)[rank % 2].contiguous()
            shape = tuple(value.shape)
            for op in gradients:
                value, shape = runtime.apply(op, value, shape, device)
            expected = torch.empty_like(weight)
            if rank == 0:
                torch.testing.assert_close(value, expected_gradient, rtol=1e-5, atol=1e-7)
                gradient_errors.append(float((value - expected_gradient).abs().max()))
                gathered = value
                value, momentum = step(weight.detach(), value, momentum)
                if args.check_dmuon:
                    from dmuon.optim.newton_schulz import direct_newton_schulz

                    ns_input = gathered + 0.95 * momentum
                    torch.testing.assert_close(orthogonalize(ns_input), direct_newton_schulz(ns_input), rtol=0, atol=0)
                expected.copy_(value)
            dist.broadcast(expected, src=0)
            shape = tuple(weight.shape)
            for op in publication:
                value, shape = runtime.apply(op, value, shape, device)
            expected_piece = expected.chunk(2, dim=0)[rank % 2]
            torch.testing.assert_close(value, expected_piece, rtol=0, atol=0)
            errors.append(float((value - expected_piece).abs().max()))
            losses.append(float(loss))
            full = expected
        with torch.no_grad():
            weight.copy_(full)
    result = dict(status="passed", mode=args.mode, steps=args.steps, topology=topology(),
                  transport=args.transport, sources=SOURCES, torch=torch.__version__,
                  transformers=transformers.__version__, losses=losses, max_update_error=max(errors),
                  max_gradient_error=max(gradient_errors, default=None),
                  dmuon_direct_ns_checked=args.check_dmuon,
                  reference_communication=["world all_reduce for gradient oracle", "world broadcast for publication oracle"] if runtime else [],
                  tiny_parameters=TINY.parameter_count(), parameter="model.layers.0.self_attn.q_proj.weight",
                  scope="real tiny Qwen gradients; one matrix's direct-Muon optimizer boundary",
                  remaining=["compose this HSDP/TP boundary with pipeline.py PP/CP/EP training",
                             "optimizer checkpoint/resume"])
    if runtime is not None:
        result.update(nccl=torch.cuda.nccl.version(), trace=runtime.trace)
        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()
    (args.output / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    if rank == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
