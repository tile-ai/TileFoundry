"""HIR collective evaluation and independent Qwen gradient production."""

from tilefoundry import core
from tilefoundry.interfaces import evaluator, typeinfer
from tilefoundry.ir import core as ir_core
from tilefoundry.ir import expr, types

from config import TINY


def collective_function(op, shape, dtype):
    """Infer the primitive once through the ordinary HIR function interface."""
    type_ = types.TensorType.from_shape(tuple(shape), dtype=evaluator.from_torch_dtype(dtype))
    source = expr.Var("payload", type=type_)
    call = expr.Call(op, (source,), type=type_)
    typeinfer.TypeInferVisitor().visit(call, typeinfer.TypeInferContext())
    return ir_core.Function.from_parts(name="communicate", params=(source,), body=call, return_type=call.type)


def apply_collective(op, payloads):
    """Execute one authored primitive and retain unrelated participant pieces."""
    members = types.Placement.from_mesh(op.participants)
    first = payloads[members[op.root or 0].ids["gpu"]]
    function = collective_function(op, first.shape, first.dtype)
    type_ = function.params[0].type
    value = core.ShardTensor(type_, tuple((types.Placement({"gpu": rank}), data) for rank, data in payloads.items()))
    result = evaluator.evaluate(function, value)
    updated = {rank: data for rank, data in payloads.items() if types.Placement({"gpu": rank}) not in members}
    updated.update((p.ids["gpu"], data) for p, data in result.tensor)
    return updated


def reference_model(device="cpu"):
    """Instantiate the scaled Qwen model from deterministic random weights."""
    import torch
    from transformers import Qwen3MoeForCausalLM

    torch.manual_seed(1234)
    return Qwen3MoeForCausalLM(TINY.transformers_config()).to(device)


def training_gradient(model, data_rank, step, tokens=8):
    """Different data ranks produce distinct gradients with equal loss-token counts."""
    import torch

    generator = torch.Generator().manual_seed(9000 + 100 * step + data_rank)
    inputs = torch.randint(TINY.vocab, (1, tokens), generator=generator).to(next(model.parameters()).device)
    model.zero_grad(set_to_none=True)
    output = model(input_ids=inputs, labels=inputs, use_cache=False)
    output.loss.backward()
    weight = model.model.layers[0].self_attn.q_proj.weight
    return output.loss.detach(), weight.grad.detach().clone()
