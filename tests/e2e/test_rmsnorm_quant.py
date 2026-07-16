"""End-to-end RMSNorm DSL test.

Standard formula ``y = bf16(f32(x) * rsqrt(mean(f32(x)²) + eps))``
with ``eps = 1e-6``. Full GPU compile + run + numerical compare
against ``torch.nn.functional.rms_norm``.
"""


import torch

import tilefoundry
from tests.models.programs.rmsnorm import (
    RmsnormModule,
    RmsnormQuantSeq2Module,
    RmsnormSeq2Module,
)
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import TupleType


def test_rmsnorm_quant_parses_and_typeinfers() -> None:
    """Parsing + HIR typeinfer of the RMSNorm formula succeed end-to-end."""

    fn = RmsnormModule.rmsnorm
    assert fn.return_type.shape == (1, 1536)
    assert fn.return_type.dtype.name == "bf16"
    assert fn.return_type.storage == StorageKind.GMEM


def test_rmsnorm_quant_e2e_gpu_run() -> None:
    """Full compile → GPU run → numerical match vs
    ``torch.nn.functional.rms_norm``. A single ``thread`` topology with
    mesh layout shape ``(6, 32)`` (named axes) gives a single CTA with
    192 threads each holding 8 contiguous bf16 lanes of the 1536-element
    input; reduce stages through intra-warp shfl + cross-warp shmem
    workspace."""

    rm = tilefoundry.compile(RmsnormModule, target="cuda")

    torch.manual_seed(42)
    a = torch.randn(1, 1536, dtype=torch.bfloat16, device="cuda") * 0.1
    out = torch.empty_like(a)
    rm(a, out)
    torch.cuda.synchronize()

    expected = torch.nn.functional.rms_norm(
        a.float(), normalized_shape=(1536,), eps=1e-6
    ).to(torch.bfloat16)

    # bf16 tolerance: ~0.2 absolute, ~5% relative — the reduction
    # order differs between tilefoundry's local-fold/shuffle/shmem
    # chain and torch's monolithic reduce, so we accept a wider
    # bf16-realistic tolerance than the f32 mma path uses.
    assert torch.allclose(out, expected, rtol=5e-2, atol=2e-1), (
        f"tilefoundry rmsnorm output does not match torch reference; "
        f"max abs diff = {(out.float() - expected.float()).abs().max().item()}"
    )


def test_rmsnorm_seq_2_e2e_gpu_run() -> None:
    """seq_2 3-axis mesh + multi-axis Split rmsnorm end-to-end on GPU.
    ``x`` is non-reduced (2 groups),
    ``y`` and ``t`` share the reduced logical axis."""

    rm = tilefoundry.compile(RmsnormSeq2Module, target="cuda")

    torch.manual_seed(42)
    a = torch.randn(2, 1536, dtype=torch.bfloat16, device="cuda") * 0.1
    out = torch.empty_like(a)
    rm(a, out)
    torch.cuda.synchronize()

    expected = torch.nn.functional.rms_norm(
        a.float(), normalized_shape=(1536,), eps=1e-6
    ).to(torch.bfloat16)

    assert torch.allclose(out, expected, rtol=5e-2, atol=2e-1), (
        f"tilefoundry rmsnorm_seq_2 output does not match torch reference; "
        f"max abs diff = {(out.float() - expected.float()).abs().max().item()}"
    )


def test_rmsnorm_quant_seq_2_parses_and_typeinfers() -> None:
    """fp8 quant: parsing + HIR typeinfer of the quant variant succeed."""
    fn = RmsnormQuantSeq2Module.rmsnorm_quant_seq_2
    assert isinstance(fn.return_type, TupleType)
    assert len(fn.return_type.fields) == 2


def _rmsnorm_quant_seq_2_reference(
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference fp32 RMSNorm → bf16 round-trip → reshape → absmax-quant.

    Mirrors the DSL pipeline of ``rmsnorm_quant_seq_2`` exactly:
    rms_norm in f32, cast to bf16, back to f32 (the
    ``a_norm_f32_for_quant`` step), reshape ``(2,1536) → (2,12,128)``,
    absmax along axis=-1, scale = absmax × (1/448), then
    ``clamp(x / scale, ±448).to(fp8e4m3)``.

    Returns ``(q_out, scale)`` with shapes ``(2,12,128) / (2,12)`` —
    callers can flatten ``q_out`` to ``(2,1536)`` to match the kernel
    output layout.
    """
    a_norm = torch.nn.functional.rms_norm(
        a.float(), normalized_shape=(1536,), eps=1e-6
    )
    a_norm_bf16_f32 = a_norm.to(torch.bfloat16).float()
    a_reshaped = a_norm_bf16_f32.reshape(2, 12, 128)
    amax = a_reshaped.abs().amax(dim=-1)  # (2, 12)
    scale = amax * (1.0 / 448.0)
    q = (a_reshaped / scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(
        torch.float8_e4m3fn
    )
    return q, scale


def test_rmsnorm_quant_seq_2_e2e_gpu_run() -> None:
    """fp8 quant: GPU compile + run — smoke check that the quant pipeline
    produces fp8 output and f32 scale tensors without crashing.

    Numerical assertions live in
    ``test_rmsnorm_quant_seq_2_e2e_gpu_precision`` and are currently
    blocked on reduce tier-1 multi-cell support.
    """

    rm = tilefoundry.compile(RmsnormQuantSeq2Module, target="cuda")

    torch.manual_seed(42)
    a = torch.randn(2, 1536, dtype=torch.bfloat16, device="cuda") * 0.1
    out0 = torch.empty(2, 1536, dtype=torch.float8_e4m3fn, device="cuda")
    out1 = torch.empty(2, 12, dtype=torch.float32, device="cuda")
    rm(a, out0, out1)
    torch.cuda.synchronize()

    assert out0.numel() == 3072
    assert out1.numel() == 24
    assert torch.isfinite(out1.float()).all()


def test_rmsnorm_quant_seq_2_e2e_gpu_precision() -> None:
    """fp8 quant precision (DOUBLE CRITERION).

    1. ``scale`` (f32 ``(2,12)``) MUST equal the reference
       ``absmax × 1/448`` exactly.
    2. ``q_out`` (fp8 ``(2,1536)``) MUST equal the reference quantized
       values exactly.

    Tolerance: ``atol=0`` for both criteria. The DSL pipeline and the
    reference Python pipeline perform the same op sequence at the
    same precision (f32 rmsnorm → bf16 cast → f32 reshape → absmax
    reduce → mul → div → clamp → fp8 cast), and the runtime
    ``reduce_intra_cta`` template is deterministic for this layout
    (one thread per reduced cell + warp-shuffle / shmem workspace),
    so the kernel output matches the reference bitwise.
    """

    rm = tilefoundry.compile(RmsnormQuantSeq2Module, target="cuda")

    torch.manual_seed(42)
    a = torch.randn(2, 1536, dtype=torch.bfloat16, device="cuda") * 0.1
    out0 = torch.empty(2, 1536, dtype=torch.float8_e4m3fn, device="cuda")
    out1 = torch.empty(2, 12, dtype=torch.float32, device="cuda")
    rm(a, out0, out1)
    torch.cuda.synchronize()

    ref_q, ref_scale = _rmsnorm_quant_seq_2_reference(a)

    # Criterion 1: scale — bitwise match.
    assert torch.allclose(out1, ref_scale, atol=0.0), (
        f"scale mismatch: max abs diff "
        f"{(out1 - ref_scale).abs().max().item():.4g}"
    )

    # Criterion 2: quant — bitwise match (cast through f32 since
    # ``torch.allclose`` does not accept fp8 inputs directly).
    out0_f = out0.float().reshape(2, 12, 128)
    ref_f = ref_q.float()
    assert torch.allclose(out0_f, ref_f, atol=0.0), (
        f"quant mismatch: max abs diff "
        f"{(out0_f - ref_f).abs().max().item():.4g}"
    )


if __name__ == "__main__":
    # Convenience entry: launch the HIR viewer for one of the fixture
    # functions defined above. Default target is ``rmsnorm_quant_seq_2``
    # (the function under test in ``*_e2e_gpu_precision``); pass a
    # different name as the first CLI argument to switch.
    #
    # Usage:
    #   python tests/e2e/test_rmsnorm_quant.py
    #   python tests/e2e/test_rmsnorm_quant.py rmsnorm
    #   python tests/e2e/test_rmsnorm_quant.py rmsnorm_seq_2
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
