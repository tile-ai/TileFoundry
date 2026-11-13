"""Compile CuTe once per worker and translate atom layouts into comparable units."""

from __future__ import annotations

import functools
import itertools
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from tilefoundry.ir import pattern, types
from tilefoundry.ir.types import int_tuple, layout_algebra
from tilefoundry_nvidia.cuda.ir.tir.nn import wgmma

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
    requests = {"mma-bf16": "mma bf16 - 0 - 0", "mma-fp8e4m3": "mma fp8e4m3 - 0 - 0", "stmatrix": "stmatrix bf16 - 0 - 0", "ldmatrix": "ldmatrix bf16 - 0 - 0"}
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
        shutil.copyfile(Path(__file__).with_name("cute_reference.cpp"), build / "reference.cpp")
        (build / "CMakeLists.txt").write_text("""
cmake_minimum_required(VERSION 3.18)
project(cute_reference LANGUAGES CXX)
find_package(CUDAToolkit REQUIRED)
find_path(CUTLASS_INCLUDE_DIR cute/tensor.hpp REQUIRED)
find_path(CCCL_INCLUDE_DIR cuda/std/cstddef HINTS ${CUDAToolkit_INCLUDE_DIRS} PATH_SUFFIXES cccl REQUIRED)
find_program(CCACHE_EXECUTABLE ccache)
add_executable(reference reference.cpp)
target_compile_features(reference PRIVATE cxx_std_17)
target_compile_definitions(reference PRIVATE CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
target_include_directories(reference PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS} ${CCCL_INCLUDE_DIR})
if(CCACHE_EXECUTABLE)
  set_target_properties(reference PROPERTIES CXX_COMPILER_LAUNCHER "${CCACHE_EXECUTABLE}")
endif()
""")
        _run(["cmake", "-G", "Ninja", "-S", str(build), "-B", str(build / "build"),
              f"-DCMAKE_CXX_COMPILER={compiler}"])
        _run(["cmake", "--build", str(build / "build")])
        executable = build / "build/reference"
        output = _run([str(executable)], input="\n".join(f"{k} {v}" for k, v in requests.items()))
    try:
        reference = json.loads(output)
    except json.JSONDecodeError as error:
        raise AssertionError(f"CuTe reference returned invalid JSON: {output}") from error
    for key in requests:
        assert key in reference, f"CuTe reference omitted configuration {key}"
    return reference


def _canonical(shape, strides, *, values=False):
    modes = list(zip(types.flatten(tuple(shape)), types.flatten(tuple(strides)), strict=True))
    if values:
        modes.sort(key=lambda pair: pair[1])
    return layout_algebra.coalesce(types.Layout(tuple(s for s, _ in modes), tuple(d for _, d in modes)))


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
            types.Layout(s, d) if broadcast and index == 1 else _canonical(s, d, values=index == 1)
        )
    return types.Layout(tuple(p.shape for p in parts), tuple(p.strides for p in parts))


def to_tv_layout(atom, role):
    """Map (thread, value) to CuTe's column-major logical operand coordinates."""
    copy = role in ("src", "dst")
    bindings = getattr(atom, "bindings", {})
    assert role in atom.roles
    operand = atom.role(role) if not copy else getattr(type(atom), role).pattern
    shape = pattern.declared_shape(pattern.selected(operand, bindings), bindings)
    declared = pattern.selected(operand, bindings).layout
    if copy and not isinstance(declared, pattern.ShardLayout):
        vector = declared.predicates[0].widths[0] * 8
        rows = int_tuple.product(shape) * 16 // vector
        return types.Layout(((rows,), (vector,)), ((vector,), (1,)))
    if declared.attrs == wgmma.BROADCAST:
        shape = shape[::-1] if role == "B" else shape
        threads = int_tuple.product(types.flatten(atom.required_execution_mesh.layout.shape))
        return types.Layout(((threads,), shape), ((0,), (1, shape[0])))

    if copy:
        mesh_pattern = type(atom).execution_mesh.pattern
        mesh = types.Mesh(
            (types.Topology("thread", 32),),
            types.Layout(
                pattern.fixed_value(mesh_pattern.layout.outer.shape, bindings),
                pattern.fixed_value(mesh_pattern.layout.outer.strides, bindings),
            ),
        )
        shard = types.ShardLayout(types.Layout(declared.layout.shape), declared.attrs, mesh)
    else:
        shard = _bound_fragment(atom, role)
    axes = int_tuple.mode_axes(int_tuple.compatible_profile(shard.layout.shape, shape))
    extents = types.flatten(shard.layout.shape)
    steps = [0] * len(extents)
    for axis in range(len(shape)):
        group = tuple(i for i, target in enumerate(axes) if target == axis)
        step = (shape[1] if axis == 0 else 1) if role == "B" else (1 if axis == 0 else shape[0])
        for position in reversed(group):
            steps[position] = step
            step *= extents[position]
    if copy:
        for axis in range(len(shape)):
            group = tuple(i for i, target in enumerate(axes) if target == axis)
            step = 128 if axis == 0 else 16
            reached = 1
            for position in reversed(group):
                steps[position] = step
                reached *= extents[position]
                step *= extents[position]
                if axis == 1 and reached == 8:
                    step = 2048
    thread_axes = [attr.axis for attr in shard.attrs]
    mesh_steps = types.flatten(shard.mesh.layout.strides)
    thread_axes = [axis for _, axis in sorted(zip(mesh_steps, thread_axes, strict=True))]
    value_axes = [i for i in range(len(steps)) if i not in thread_axes]
    parts = [
        _canonical(
            ((16,) if copy and index == 1 else ()) + tuple(extents[i] for i in axes),
            ((1,) if copy and index == 1 else ()) + tuple(steps[i] for i in axes),
            values=index == 1,
        )
        for index, axes in enumerate((thread_axes, value_axes))
    ]
    return types.Layout(tuple(p.shape for p in parts), tuple(p.strides for p in parts))


def to_byte_layout(layout, shape, element_bytes, region):
    """Enumerate a rectangular region; swizzle acts on bytes, including the offset."""
    grouped = types.unflatten(types.flatten(layout.outer), int_tuple.compatible_profile(layout.outer.shape, shape))
    addresses = []
    for row, column in itertools.product(range(region[0]), range(region[1])):
        offset = layout.offset + types.crd2idx((row, column), grouped.shape, grouped.strides)
        addresses.append(layout.inner(offset * element_bytes))
    return tuple(addresses)


def to_thread_layout(atom):
    """Read the declared execution arrangement in increasing physical-thread order."""
    if hasattr(atom, "role"):
        layout = atom.required_execution_mesh.layout
    else:
        assert pattern.declared_execution_mesh(atom).match(atom.required_execution_mesh) is not None
        declared = type(atom).execution_mesh.pattern.layout.outer
        layout = types.Layout(declared.shape, declared.strides)
    modes = sorted(
        zip(types.flatten(layout.shape), types.flatten(layout.strides), strict=True), key=lambda p: p[1]
    )
    return layout_algebra.coalesce(types.Layout(tuple(s for s, _ in modes), tuple(d for _, d in modes)))


def reference_threads(raw):
    """Canonicalize CuTe's thread numbering as a row-major layout."""
    shape, strides = raw["shape"], raw["strides"]
    return _canonical(
        shape if isinstance(shape, list) else (shape,),
        strides if isinstance(strides, list) else (strides,),
        values=True,
    )
