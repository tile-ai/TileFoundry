"""End-to-end RMSNorm DSL test.

Standard formula ``y = bf16(f32(x) * rsqrt(mean(f32(x)²) + eps))``
with ``eps = 1e-6``. Full GPU compile + run + numerical compare
against ``torch.nn.functional.rms_norm``.
"""

import pytest
import torch

from tests.fixtures.placed.rmsnorm import RmsnormModule
from tests.fixtures.placed.rmsnorm_quant_seq2 import RmsnormQuantSeq2Module
from tests.fixtures.placed.rmsnorm_seq2 import RmsnormSeq2Module

NORMALISED = [
    pytest.param(RmsnormModule, 1, id="one_reduced_axis"),
    pytest.param(RmsnormSeq2Module, 2, id="seq_2_multi_axis_split"),
]


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
