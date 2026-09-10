"""The runtime's compile-time refusals, read as diagnostics.

See [runtime §3](docs/spec/runtime.md#3-runtime-ops).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = Path(__file__).resolve().parent / "cuda" / "static_asserts.cu"

_INCLUDES = ("-I", str(_ROOT / "include"), "-I", str(_ROOT / "third_party" / "cutlass" / "include"))

CASES: dict[int, str | None] = {
    0: None,
    1: "a mesh layout's first component must be cute::identity",
    2: "a mesh layout's offset must be a compile-time number",
    3: "a shard layout must say what each of its mesh's axes does with the tensor",
    4: "make_shard_layout: one attr per mesh axis",
    5: "one attr per mesh axis: this layout has one stride per attr",
    6: "this attr neither names a tensor axis",
    7: "these shard layouts spread one value over more than one warp",
    8: "ops::reduce (intra-warp tier): this reduce mesh spreads one value",
    9: "the element type must be one float holds exactly",
    10: "the source's projected layout must be a plain cute::Layout",
    11: "the two operands must name one mesh",
    12: "ops::reduce: a reduce mesh's scope must be cta or thread",
    13: "ops::reduce (intra-CTA tier): kept axes must form one cell",
    14: "these two views share no run wide enough for cp.async",
    15: "the two projected slices must hold the same number of elements",
    16: "ops::dot (warp tier): the fastest axis of the operands' mesh",
    17: "ops::dot (block tier): the operands' mesh must be a whole number of warps",
    18: "the operands are neither a rank-2 static tile",
    19: "ops::sync: a mesh's scope must be cta or thread",
    20: "ops::sync: a CTA mesh needs the module's grid-barrier counter",
    21: "ops::mma (tile tier): the accumulator's mesh must be a whole number of warps",
    22: "a reduced mesh axis must divide into whole lanes and whole warps",
    24: "both operands must leave the tile whole on every ",
    25: "ops::sync: a mesh that skips warps names no barrier",
    26: "get<level>: this mesh does not name that level",
}

pytestmark = pytest.mark.skipif(
    shutil.which("nvcc") is None, reason="the refusals are nvcc diagnostics"
)


def _compile(case: int) -> subprocess.CompletedProcess:
    """Compile the one translation unit at ``-DCASE=case``."""
    return subprocess.run(
        [
            "nvcc",
            "-std=c++20",
            "-arch=sm_90a",
            f"-DCASE={case}",
            *_INCLUDES,
            "-c",
            "-o",
            "/dev/null",
            str(_SOURCE),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_the_corrected_arithmetic_still_compiles() -> None:
    """The positive control, so that failing everywhere cannot read as passing."""
    proc = _compile(0)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("case", [c for c in CASES if c != 0])
def test_the_wrong_usage_does_not_compile(case: int) -> None:
    """Each case violates one constraint and is refused by its own sentence."""
    proc = _compile(case)
    assert proc.returncode != 0, f"CASE {case} compiled; the constraint is not enforced"
    expected = CASES[case]
    assert expected is not None
    assert expected in proc.stderr, (
        f"CASE {case} failed for a different reason than the constraint it pins;\n"
        f"expected to find: {expected!r}\nstderr:\n{proc.stderr}"
    )
