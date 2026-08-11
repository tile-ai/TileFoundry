"""End-to-end RMSNorm DSL test.

Standard formula ``y = bf16(f32(x) * rsqrt(mean(f32(x)²) + eps))``
with ``eps = 1e-6``. Full GPU compile + run + numerical compare
against ``torch.nn.functional.rms_norm``.
"""

import pytest
import torch

import tilefoundry
from tests.fixtures.placed.rmsnorm import RmsnormModule
from tests.fixtures.placed.rmsnorm_quant_seq2 import RmsnormQuantSeq2Module
from tests.fixtures.placed.rmsnorm_seq2 import RmsnormSeq2Module
from tilefoundry.target import CudaTarget

NORMALISED = [
    pytest.param(RmsnormModule, 1, id="one_reduced_axis"),
    pytest.param(RmsnormSeq2Module, 2, id="seq_2_multi_axis_split"),
]


@pytest.mark.parametrize(("normaliser", "batch"), NORMALISED)
def test_a_compiled_rmsnorm_matches_torch(normaliser, batch) -> None:
    """Full compile → GPU run → numerical match vs ``torch.nn.functional.rms_norm``.

    bf16 tolerance is ~0.2 absolute, ~5% relative: the reduction order differs
    between tilefoundry's local-fold/shuffle/shmem chain and torch's monolithic
    reduce, so this accepts a wider bf16-realistic tolerance than the f32 mma path
    uses.
    """
    rm = tilefoundry.compile(normaliser, target=CudaTarget("nvidia.h200_sxm"))

    torch.manual_seed(42)
    a = torch.randn(batch, 1536, dtype=torch.bfloat16, device="cuda") * 0.1
    out = torch.empty_like(a)
    rm(a, out)
    torch.cuda.synchronize()

    expected = torch.nn.functional.rms_norm(a.float(), normalized_shape=(1536,), eps=1e-6).to(
        torch.bfloat16
    )

    assert torch.allclose(out, expected, rtol=5e-2, atol=2e-1), (
        f"tilefoundry rmsnorm output does not match torch reference; "
        f"max abs diff = {(out.float() - expected.float()).abs().max().item()}"
    )


def _rmsnorm_quant_seq_2_reference(
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference fp32 RMSNorm → bf16 round-trip → reshape → absmax-quant.

    Mirrors the DSL pipeline exactly: f32 RMSNorm, bf16 round-trip, reshape to
    ``(2,12,128)``, last-axis absmax, scale by ``1/448``, clamp, and fp8 cast.
    Returns ``q_out`` and scale with shapes ``(2,12,128)`` and ``(2,12)``;
    callers flatten ``q_out`` to match the kernel layout.
    """
    a_norm = torch.nn.functional.rms_norm(a.float(), normalized_shape=(1536,), eps=1e-6)
    a_norm_bf16_f32 = a_norm.to(torch.bfloat16).float()
    a_reshaped = a_norm_bf16_f32.reshape(2, 12, 128)
    amax = a_reshaped.abs().amax(dim=-1)
    scale = amax * (1.0 / 448.0)
    q = (a_reshaped / scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, scale


def test_rmsnorm_quant_seq_2_e2e_gpu_precision() -> None:
    """fp8 quant precision (DOUBLE CRITERION).

    Both f32 scale ``absmax / 448`` and fp8 output must equal the reference
    exactly. DSL and Python execute the same precision sequence, and this
    ``reduce_intra_cta`` layout is deterministic, so tolerance remains zero.
    """
    rm = tilefoundry.compile(RmsnormQuantSeq2Module, target=CudaTarget("nvidia.h200_sxm"))

    torch.manual_seed(42)
    a = torch.randn(2, 1536, dtype=torch.bfloat16, device="cuda") * 0.1
    out0 = torch.empty(2, 1536, dtype=torch.float8_e4m3fn, device="cuda")
    out1 = torch.empty(2, 12, dtype=torch.float32, device="cuda")
    rm(a, out0, out1)
    torch.cuda.synchronize()

    ref_q, ref_scale = _rmsnorm_quant_seq_2_reference(a)

    assert torch.allclose(out1, ref_scale, atol=0.0), (
        f"scale mismatch: max abs diff {(out1 - ref_scale).abs().max().item():.4g}"
    )

    out0_f = out0.float().reshape(2, 12, 128)
    ref_f = ref_q.float()
    assert torch.allclose(out0_f, ref_f, atol=0.0), (
        f"quant mismatch: max abs diff {(out0_f - ref_f).abs().max().item():.4g}"
    )


if __name__ == "__main__":
    import sys

    from tilefoundry.inspection.viewer import Viewer

    target_name = sys.argv[1] if len(sys.argv) > 1 else "rmsnorm_quant_seq_2"
    _fns = {
        "rmsnorm": RmsnormModule.rmsnorm,
        "rmsnorm_seq_2": RmsnormSeq2Module.rmsnorm_seq_2,
        "rmsnorm_quant_seq_2": RmsnormQuantSeq2Module.rmsnorm_quant_seq_2,
    }
    fn_obj = _fns.get(target_name)
    if fn_obj is None or not hasattr(fn_obj, "params"):
        print(
            f"unknown function {target_name!r}; available: "
            "rmsnorm, rmsnorm_seq_2, rmsnorm_quant_seq_2",
            file=sys.stderr,
        )
        sys.exit(1)
    Viewer(fn_obj).serve(port=0, open_browser=True)
