"""Compile CuTe once per worker and translate atom layouts into comparable units."""

from __future__ import annotations

import functools
import itertools
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from tilefoundry.ir.pattern.utils import selected_pattern
from tilefoundry.ir.tir.cuda.nn import wgmma
from tilefoundry.ir.types import Layout
from tilefoundry.ir.types.int_tuple import flatten, product
from tilefoundry.ir.types.layout_algebra import coalesce
from tilefoundry.ir.types.layout_correspondence import layout_correspondence

from ._operand_layout import _bound_fragment


def tv_configs():
    """Request every supported WGMMA dtype, form and N."""
    return itertools.product(wgmma.DTYPES, tuple(wgmma.Form), range(8, 257, 8))


def smem_atoms():
    """Request each legal shared-memory atom independently of N."""
    return (
        (dtype, major, swizzle)
        for dtype in wgmma.DTYPES
        for major in ("MN", "K")
        if major == "K" or dtype.bit_width == 16
        for swizzle in range(4)
    )


def tv_key(dtype, form, n):
    return f"tv-{dtype.name}-{form.value}-{n}"


def smem_key(dtype, major, swizzle):
    return f"smem-{dtype.name}-{major}-{swizzle}"


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, f"{command}:\n{result.stderr}"
    return result.stdout


@functools.cache
def cute_reference():
    """Return JSON from precisely the configurations sent to the CuTe executable."""
    compiler = shutil.which("g++")
    assert compiler, "CuTe reference requires g++ in PATH"
    root = Path(__file__).resolve().parents[4]
    cutlass = root / "third_party/cutlass/include"
    nvcc = shutil.which("nvcc")
    cuda = (
        Path(os.environ["CUDA_HOME"]) / "include"
        if os.environ.get("CUDA_HOME")
        else Path(nvcc).resolve().parent.parent / "include"
        if nvcc
        else None
    )
    assert (cutlass / "cute/tensor.hpp").is_file(), f"Missing CuTe headers: {cutlass}"
    assert cuda and (cuda / "cuda_runtime_api.h").is_file(), (
        f"Missing CUDA headers: searched CUDA_HOME/include and nvcc/../include ({cuda})"
    )
    requests = {"mma": "mma bf16 - 0 - 0"}
    requests.update(
        {
            tv_key(dtype, form, n): f"tv {dtype.name} {form.value} {n} K 0"
            for dtype, form, n in tv_configs()
        }
    )
    requests.update(
        {
            smem_key(dtype, major, sw): f"smem {dtype.name} - 0 {major} {sw}"
            for dtype, major, sw in smem_atoms()
        }
    )
    with tempfile.TemporaryDirectory(prefix="tilefoundry_cute_reference_") as directory:
        build = Path(directory)
        obj, executable = build / "reference.o", build / "reference"
        ccache = shutil.which("ccache")
        _run(
            [
                *([ccache] if ccache else []),
                compiler,
                "-std=c++17",
                "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
                f"-I{cutlass}",
                f"-I{cuda}",
                f"-I{cuda}/cccl",
                "-c",
                str(Path(__file__).with_name("cute_reference.cpp")),
                "-o",
                str(obj),
            ]
        )
        _run([compiler, str(obj), "-o", str(executable)])
        output = _run([str(executable)], input="\n".join(f"{k} {v}" for k, v in requests.items()))
    try:
        reference = json.loads(output)
    except json.JSONDecodeError as error:
        raise AssertionError(f"CuTe reference returned invalid JSON: {output}") from error
    for key in requests:
        assert key in reference, f"CuTe reference omitted configuration {key}"
    return reference


def _canonical(shape, strides, *, values=False):
    modes = list(zip(flatten(tuple(shape)), flatten(tuple(strides)), strict=True))
    if values:
        modes.sort(key=lambda pair: pair[1])
    return coalesce(Layout(tuple(s for s, _ in modes), tuple(d for _, d in modes)))


def reference_tv(raw, *, broadcast=False):
    """Preserve broadcast operand axes; canonicalize register element sets."""

    def tuples(value):
        return tuple(tuples(x) for x in value) if isinstance(value, list) else value

    shape, strides = tuples(raw["shape"]), tuples(raw["strides"])
    parts = []
    for index in range(2):
        s, d = shape[index], strides[index]
        s = s if isinstance(s, tuple) else (s,)
        d = d if isinstance(d, tuple) else (d,)
        parts.append(
            Layout(s, d) if broadcast and index == 1 else _canonical(s, d, values=index == 1)
        )
    return Layout(tuple(p.shape for p in parts), tuple(p.strides for p in parts))


def to_tv_layout(atom, role):
    """Map (thread, value) to CuTe's column-major logical operand coordinates."""
    shape = atom.operand_shapes()[("C", "A", "B").index(role)]
    pattern = selected_pattern(atom.role(role), atom.bindings).layout
    if pattern.attrs == wgmma.BROADCAST:
        shape = shape[::-1] if role == "B" else shape
        threads = product(flatten(atom.required_execution_mesh.layout.shape))
        return Layout(((threads,), shape), ((0,), (1, shape[0])))

    shard = _bound_fragment(atom, role)
    corresponding = layout_correspondence(shard.layout.shape, shape)
    steps = [0] * len(corresponding.extents)
    for axis, group in enumerate(corresponding.groups):
        step = (shape[1] if axis == 0 else 1) if role == "B" else (1 if axis == 0 else shape[0])
        for position in reversed(group):
            steps[position] = step
            step *= corresponding.extents[position]
    thread_axes = [attr.axis for attr in shard.attrs]
    mesh_steps = flatten(shard.mesh.layout.strides)
    thread_axes = [axis for _, axis in sorted(zip(mesh_steps, thread_axes, strict=True))]
    value_axes = [i for i in range(len(steps)) if i not in thread_axes]
    parts = [
        _canonical(
            tuple(corresponding.extents[i] for i in axes),
            tuple(steps[i] for i in axes),
            values=index == 1,
        )
        for index, axes in enumerate((thread_axes, value_axes))
    ]
    return Layout(tuple(p.shape for p in parts), tuple(p.strides for p in parts))


def to_byte_layout(layout, shape, element_bytes, region):
    """Enumerate a rectangular region; swizzle acts on bytes, including the offset."""
    corresponding = layout_correspondence(layout.outer.shape, shape)
    strides = flatten(layout.outer.strides)
    addresses = []
    for row, column in itertools.product(range(region[0]), range(region[1])):
        digits = corresponding.digits(row * shape[1] + column)
        offset = layout.offset + sum(x * d for x, d in zip(digits, strides, strict=True))
        addresses.append(layout.inner(offset * element_bytes))
    return tuple(addresses)
