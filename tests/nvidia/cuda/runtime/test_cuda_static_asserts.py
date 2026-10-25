"""The runtime's compile-time refusals, read as diagnostics.

See [runtime §2.6](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/runtime.md#26-cudaops).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tilefoundry_nvidia import CudaTarget

_ROOT = Path(__file__).resolve().parents[4]
_SOURCE = Path(__file__).resolve().parent / "cuda" / "static_asserts.cu"

_INCLUDES = ("-I", str(_ROOT / "targets/nvidia/tilefoundry_nvidia/cuda/codegen/include"), "-I", str(_ROOT / "third_party" / "cutlass" / "include"))

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
    14: "these two views share no run wide enough for cp.async",
    15: "the two projected slices must hold the same number of elements",
    16: "ops::dot (warp tier): the fastest axis of the operands' mesh",
    17: "ops::dot (block tier): the operands' mesh must be a whole number of warps",
    18: "operands must be one atom's lane fragments",
    19: "ops::sync: a mesh's scope must be cta or thread",
    20: "ops::sync: a CTA mesh needs the module's grid-barrier counter",
    22: "a reduced mesh axis must divide into whole lanes and whole warps",
    24: "ops::copy_async_bulk: both operands must leave the tile whole on every ",
    25: "ops::sync: a mesh that skips warps names no barrier",
    26: "get<level>: this mesh does not name that level",
    27: "operands must be one atom's lane fragments",
    28: "operands must be one atom's lane fragments",
    29: "operands must be one atom's lane fragments",
}

_DIAGNOSTIC = re.compile(r"\n(?=\S+\(\d+\): (?!note))")
_ERROR = re.compile(r"(\S+)\((\d+)\): (?:catastrophic )?error")
_CONTEXT = re.compile(r" at line (\d+)(?: of (\S+))?$", re.M)
_CASE_BLOCK = re.compile(r"^#if CASE == (\d+) \|\| CASE == ALL$.*?^#endif$", re.M | re.S)

pytestmark = pytest.mark.skipif(
    shutil.which("nvcc") is None, reason="the refusals are nvcc diagnostics"
)


def _compile(case: int | str, arch: str = "sm_90a") -> subprocess.CompletedProcess:
    """Compile the one translation unit at ``-DCASE=case`` for *arch*."""
    return subprocess.run(
        [
            "nvcc",
            "-std=c++20",
            f"-arch={arch}",
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


def _case_lines() -> dict[int, range]:
    """The lines of each refused case's ``#if`` block in the ``.cu``."""
    text = _SOURCE.read_text()
    lines = {}
    for block in _CASE_BLOCK.finditer(text):
        first = text.count("\n", 0, block.start()) + 1
        lines[int(block[1])] = range(first, first + block[0].count("\n") + 1)
    return lines


def _source_line(error: str) -> int | None:
    """The ``.cu`` line of *error*'s outermost instantiation context, else of *error* itself."""
    file, line = _ERROR.match(error).groups()
    *_, (line, of) = [(line, file), *_CONTEXT.findall(error)]
    return int(line) if Path(of or file).name == _SOURCE.name else None


@pytest.mark.parametrize("arch", ["sm_90a", CudaTarget("nvidia.rtx_6000_ada").arch])
def test_static_asserts_baseline(arch: str) -> None:
    """The positive control, so that failing everywhere cannot read as passing.

    On sm_89 it also shows the runtime headers compile for an architecture
    without the sm_90 instructions they hold in templates this unit leaves
    uninstantiated.
    """
    proc = _compile(0, arch)
    assert proc.returncode == 0, proc.stderr


def test_static_asserts_refuse() -> None:
    """Each case violates one constraint and is refused by its own sentence.

    All refused cases compile in one unit. A diagnostic starts at column 0 as
    ``file(line): kind`` and keeps its notes; it belongs to the case whose ``#if``
    block holds its outermost ``at line N`` (a context without ``of file`` is in
    the diagnostic's own file), and an error of no case fails.
    """
    proc = _compile("ALL")
    case_lines = _case_lines()
    errors: dict[int, list[str]] = {case: [] for case, sentence in CASES.items() if sentence}
    problems = []
    for error in filter(_ERROR.match, _DIAGNOSTIC.split(proc.stderr)):
        line = _source_line(error)
        owner = next((case for case, lines in case_lines.items() if line in lines), None)
        if owner in errors:
            errors[owner].append(error)
        else:
            problems.append(f"an error of no refused case (line {line}):\n{error}")
    for case, owned in errors.items():
        if not any(CASES[case] in error for error in owned):
            heads = "".join(f"\n  {error.splitlines()[0]}" for error in owned) or " none"
            problems.append(f"CASE {case} is not refused by {CASES[case]!r}; its errors:{heads}")
    assert not problems, "\n\n".join(problems)
