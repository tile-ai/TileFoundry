"""Feature-map geometry agrees across singleton layouts and padded storage."""

import pytest

from tilefoundry_apple.ane import isa


@pytest.mark.parametrize("dtype", ("f16", "f32"))
@pytest.mark.parametrize("shape", ((1, 32, 1, 1), (1, 1, 3, 16), (1, 1, 1, 1), (2, 1, 1, 3)))
def test_singleton_strides_preserve_geometry(dtype, shape):
    canonical = isa.TensorDesc.from_shape(dtype, shape)
    strides = tuple(7 if extent == 1 else step for extent, step in zip(shape, canonical.strides))
    desc = isa.TensorDesc(dtype, shape, strides)

    desc.validate()
    assert desc.is_canonical
    assert desc.byte_strides == canonical.byte_strides
    assert desc.storage_bytes == canonical.canonical_geometry[3]
    assert desc.declared_geometry == canonical.canonical_geometry


@pytest.mark.parametrize("strides", ((0, 1024, 32, 1), (0, 512, 64, 1), (0, 512, 32, 2)))
def test_noncanonical_strides_are_rejected(strides):
    desc = isa.TensorDesc("f16", (1, 32, 16, 16), strides)
    assert not desc.is_canonical
    with pytest.raises(ValueError, match="not canonical"):
        desc.validate()


@pytest.mark.parametrize("strides", ((0, 512, 0, 0), (0, 512, 32, 1)))
def test_declared_singleton_rows_keep_channel_padding(strides):
    desc = isa.TensorDesc("f16", (1, 32, 1, 1), strides)
    desc.validate("declared")
    assert desc.declared_geometry == (64, 1024, 32768, 32768)
    with pytest.raises(ValueError, match="not canonical"):
        desc.validate()


def test_tile_pitch_preserves_dram_geometry():
    desc = isa.TensorDesc.from_shape("f16", (1, 32, 16, 64))
    assert desc.staging_channel(16) == 32
    assert desc.staging_row(isa.Engine.PLANAR, width=16) == 1056
    assert desc.declared_geometry == (128, 2048, 65536, 65536)
    with pytest.raises(ValueError, match="tile_width"):
        desc.staging_channel(65)


@pytest.mark.parametrize(
    ("dtype", "shape", "strides", "error"),
    (
        ("i32", (1, 32, 1, 1), (0, 32, 0, 0), "dtype"),
        ("f16", (1, 32, 0, 1), (0, 32, 0, 0), "extents"),
        ("f16", (1, 32, 1, 1), (0, None, 0, 0), "strides"),
        ("f16", (1, 32, 16, 16), (0, 256, 32, 1), "cover"),
        ("f16", (2, 32, 16, 16), (1, 512, 32, 1), "batch stride"),
    ),
)
def test_invalid_declared_layouts_are_rejected(dtype, shape, strides, error):
    with pytest.raises(ValueError, match=error):
        isa.TensorDesc(dtype, shape, strides).validate("declared")
