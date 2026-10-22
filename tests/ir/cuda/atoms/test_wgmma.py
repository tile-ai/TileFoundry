"""SM90 operand ownership and shared-memory atom addresses compared with CuTe."""

from tilefoundry.dsl import T
from tilefoundry.ir.tir.cuda.nn import wgmma
from tilefoundry.ir.types import ComposedLayout, Layout, Swizzle

from ._cute_reference import (
    cute_reference,
    reference_tv,
    smem_atoms,
    smem_key,
    to_byte_layout,
    to_tv_layout,
    tv_configs,
    tv_key,
)
from ._operand_layout import (
    _assert_reshard_typeinfer_ok,
    _bound_fragment,
    _check_split_extents_match_mesh,
)


def _atom_slices(dtype, major, swizzle, role, raw):
    """Instantiate and match the minimal pattern, then select each CuTe atom slice."""
    eb = dtype.bit_width // 8
    width = wgmma.SWIZZLE_BYTES[swizzle] // eb
    issue_k = 32 // eb
    extent = 64 if role == "A" else width if major == "MN" else 8
    leading = 8 * width
    if major == "MN":
        suffix = "mn_major"
        shape = ((extent // width, width), (2, 8))
        strides = ((leading, 1), (leading, width))
    elif swizzle <= 1:
        suffix = "k_major_whole"
        shape = ((extent // 8, 8), (issue_k // width, width))
        strides = ((leading, width), (leading, 1))
    else:
        suffix = "k_major_sliced"
        shape = ((extent // 8, 8), issue_k)
        strides = ((leading, width), 1)
    major_mode = wgmma.Major.MN if major == "MN" else wgmma.Major.K
    pattern = wgmma._smem_pattern(eb, major_mode, b_operand=role == "B")
    ref_shape = tuple(raw["layout"]["shape"])
    ref_strides = tuple(raw["layout"]["strides"])
    offsets = range(0, ref_shape[1], issue_k) if suffix == "k_major_sliced" else (0,)
    for offset in offsets:
        ours = ComposedLayout(
            Swizzle(swizzle, 4, 3),
            offset,
            Layout(
                shape if role == "A" else shape[::-1], strides if role == "A" else strides[::-1]
            ),
        )
        canonical = wgmma.canonical_smem_layout(ours, eb, major_mode, b_operand=role == "B")
        assert pattern.match(canonical, {"n": extent}) is not None, (
            dtype,
            major,
            swizzle,
            role,
            offset,
            ours,
        )
        region = (ref_shape[0], min(issue_k, ref_shape[1]))
        theirs = ComposedLayout(
            Swizzle(*raw["swizzle"]),
            offset * ref_strides[1],
            Layout(
                region if role == "A" else region[::-1],
                ref_strides if role == "A" else ref_strides[::-1],
            ),
        )
        logical = (extent, issue_k)
        yield (
            offset,
            ours,
            theirs,
            logical if role == "A" else logical[::-1],
            region if role == "A" else region[::-1],
        )


def test_wgmma_atom():
    """All supported TV layouts and shared-memory atom slices agree with CuTe."""
    reference = cute_reference()
    failures = []
    for dtype, form, n in tv_configs():
        key = tv_key(dtype, form, n)
        atom = T.cuda.sm90.Wgmma(
            n=n,
            dtype=dtype,
            form=form,
            a_major=wgmma.Major.K,
            b_major=wgmma.Major.K,
        )
        for role in ("A", "B", "C"):
            broadcast = role == "B" or role == "A" and form == wgmma.Form.SS
            if not broadcast:
                shard = _bound_fragment(atom, role)
                _check_split_extents_match_mesh(shard)
                shape = atom.operand_shapes()[("C", "A", "B").index(role)]
                _assert_reshard_typeinfer_ok(shape, "f32" if role == "C" else dtype.name, shard)
            ours = to_tv_layout(atom, role)
            theirs = reference_tv(reference[key][role], broadcast=broadcast)
            if ours != theirs:
                failures.append(f"{key} {role}: {ours} != {theirs}")
    for dtype, major, swizzle in smem_atoms():
        key = smem_key(dtype, major, swizzle)
        eb = dtype.bit_width // 8
        for role in ("A", "B"):
            for offset, ours, theirs, shape, region in _atom_slices(
                dtype,
                major,
                swizzle,
                role,
                reference[key],
            ):
                actual = to_byte_layout(ours, shape, eb, region)
                expected = to_byte_layout(theirs, region, eb, region)
                if actual != expected:
                    first = next(
                        i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]
                    )
                    failures.append(
                        f"{key} {role} offset={offset}: {ours} != {theirs}; "
                        f"byte address at point {first}: {actual[first]} != {expected[first]}"
                    )
    assert not failures, "\n".join(failures)
