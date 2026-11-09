"""Programs whose shared memory accesses race without their sync steps or order, on an sm_89 card.

Each finalizes through the default flow for the card and must match its reference
exactly.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

import tilefoundry
from tilefoundry import analysis, cli, runtime
from tilefoundry.interfaces import evaluator
from tilefoundry.schedule import finalize
from tilefoundry_nvidia import cuda

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _gemm(function):
    """Integer-valued e4m3 operands, whose partial sums are exact."""
    generator = torch.Generator().manual_seed(0)
    a = torch.randint(-4, 5, (16, 128), generator=generator).float().to(torch.float8_e4m3fn)
    weight = torch.randint(-4, 5, (8, 128), generator=generator).float().to(torch.float8_e4m3fn)
    return (a, weight.t()), a.float() @ weight.float().t()


def _identity(function):
    (param,) = function.params
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(tuple(param.type.shape), generator=generator).to(evaluator.to_torch_dtype(param.type.dtype))
    return (x,), x


def _evaluated(function):
    generator = torch.Generator().manual_seed(0)
    inputs = tuple(
        torch.randn(tuple(param.type.shape), generator=generator).to(
            evaluator.to_torch_dtype(param.type.dtype)
        )
        for param in function.params
    )
    return inputs, evaluator.evaluate(function, *inputs)


def _poisoned(param):
    dtype = evaluator.to_torch_dtype(param.type.dtype)
    fill = float("nan") if dtype.is_floating_point else -1
    return torch.full(tuple(param.type.shape), fill, dtype=dtype, device="cuda")


@pytest.mark.sm89
@pytest.mark.parametrize(
    ("path", "reference"),
    [
        ("schedule/hir/sm89_e4m3_gemm.py", _gemm),
        ("schedule/hir/submesh_slice_in_sibling.py", _evaluated),
        ("analysis/sync_nested_subsets.py", _evaluated),
        ("passes/sync_row_windows.py", _identity),
        ("passes/sync_lane_rows.py", _identity),
        ("passes/destination_order.py:Sibling", _evaluated),
        ("passes/destination_order.py:KeptAcross", _evaluated),
    ],
    ids=lambda value: Path(value).name.replace(".py", "") if isinstance(value, str) else None,
)
def test_synchronised_program_matches_on_the_card(path, reference):
    module = replace(cli.load_authored_ir(str(FIXTURES / path)), target=cuda.Target("nvidia.rtx_6000_ada"))
    inputs, expected = reference(module.entry_function())
    lowered = finalize(
        analysis.analyze(*analysis.check_program(module, module.entry_function()), analysis=("memory", "sync"))
    )
    compiled = tilefoundry.build(lowered)
    scratch = tuple(_poisoned(param) for param in compiled.type.input_params[len(inputs) :])
    outputs = tuple(_poisoned(param) for param in compiled.type.output_params)
    compiled(*(value.cuda() for value in inputs), *scratch, *outputs)
    torch.cuda.synchronize()
    actual = outputs[0] if len(outputs) == 1 else outputs
    for (_, got), (_, want) in zip(runtime.flatten_outputs(actual), runtime.flatten_outputs(expected), strict=True):
        assert torch.equal(got.cpu(), want.cpu())
