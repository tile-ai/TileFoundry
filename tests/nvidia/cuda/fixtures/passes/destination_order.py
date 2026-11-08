"""Schedules that overwrite data in place while another region or the result still needs it."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

TARGET = CudaTarget("nvidia.rtx_6000_ada")
TOPOLOGIES = (Topology("cta", 1), Topology("thread", 64))
TILE = Tensor[(32,), "f32", Layout((32,), (1,)), "smem"]
WIDE = Tensor[(64,), "f32", Layout((64,), (1,)), "smem"]


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class Sibling:
    """Each iteration a reader inserts the carried tile, then a loader copies a row into it."""

    @func
    def run(a: Tensor[(4, 32), "f32"]) -> Tensor[(4, 32), "f32"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(2, 32), names=("warp", "lane")) as threads:
                with threads[0, :32] as _loader:
                    tile = tf.zeros(TILE)
                out = tf.zeros(Tensor[(4, 32), "f32"])
                for k in range(4):
                    with threads[1, :32] as _reader:
                        out = tf.insert_slice(out, tf.reshape(tile, (1, 32)), (k, 0))
                    with threads[0, :32] as _loader:
                        tile = tf.schedule((a[k, :], tile), op=T.copy())
                return out


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class MaybeTrip:
    """The inner loop runs i rounds, so at i = 0 the read goes through the old tile."""

    @func
    def run(
        a: Tensor[(2, 32), "f32"],
    ) -> tuple[Tensor[(1, 32), "f32"], Tensor[(2, 32), "f32"]]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(2, 32), names=("warp", "lane")) as threads:
                with threads[0, :32] as _loader:
                    tile = tf.schedule((a[0, :], tf.zeros(TILE)), op=T.copy())
                out = tf.zeros(Tensor[(2, 32), "f32"])
                for i in range(2):
                    kept = tile
                    for _j in range(i):
                        with threads[0, :32] as _loader:
                            kept = tf.schedule((a[1, :], kept), op=T.copy())
                    with threads[1, :32] as _reader:
                        out = tf.insert_slice(out, tf.reshape(kept, (1, 32)), (i, 0))
                with threads[0, :32] as _loader:
                    fresh = tf.schedule((a[1, :], tile), op=T.copy())
                with threads[1, :32] as _reader:
                    new = tf.insert_slice(
                        tf.zeros(Tensor[(1, 32), "f32"]), tf.reshape(fresh, (1, 32)), (0, 0)
                    )
                return new, out


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class CaptureHandoff:
    """The carry takes a tile the loop captures each round; that tile is overwritten later."""

    @func
    def run(
        a: Tensor[(3, 32), "f32"],
    ) -> tuple[Tensor[(1, 32), "f32"], Tensor[(2, 32), "f32"]]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(2, 32), names=("warp", "lane")) as threads:
                with threads[0, :32] as _loader:
                    tile = tf.schedule((a[0, :], tf.zeros(TILE)), op=T.copy())
                with threads[0, :32] as _loader:
                    other = tf.schedule((a[1, :], tf.zeros(TILE)), op=T.copy())
                out = tf.zeros(Tensor[(2, 32), "f32"])
                kept = tile
                for k in range(2):
                    with threads[1, :32] as _reader:
                        out = tf.insert_slice(out, tf.reshape(kept, (1, 32)), (k, 0))
                        kept = tf.reshape(tf.reshape(other, (1, 32)), (32,))
                with threads[0, :32] as _loader:
                    fresh = tf.schedule((a[2, :], other), op=T.copy())
                with threads[1, :32] as _reader:
                    new = tf.insert_slice(
                        tf.zeros(Tensor[(1, 32), "f32"]), tf.reshape(fresh, (1, 32)), (0, 0)
                    )
                return new, out


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class KeptAcross:
    """A loop carries the tile on; the tile is overwritten after it, and one reader takes both."""

    @func
    def run(
        a: Tensor[(2, 32), "f32"],
    ) -> tuple[Tensor[(2, 32), "f32"], Tensor[(2, 32), "f32"]]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(2, 32), names=("warp", "lane")) as threads:
                with threads[0, :32] as _loader:
                    tile = tf.schedule((a[0, :], tf.zeros(TILE)), op=T.copy())
                out = tf.zeros(Tensor[(2, 32), "f32"])
                kept = tile
                for k in range(2):
                    with threads[1, :32] as _reader:
                        out = tf.insert_slice(out, tf.reshape(kept, (1, 32)), (k, 0))
                        kept = tf.reshape(tf.reshape(kept, (1, 32)), (32,))
                with threads[0, :32] as _loader:
                    fresh = tf.schedule((a[1, :], tile), op=T.copy())
                with threads[1, :32] as _reader:
                    both = tf.insert_slice(
                        tf.zeros(Tensor[(2, 32), "f32"]), tf.reshape(fresh, (1, 32)), (1, 0)
                    )
                    both = tf.insert_slice(both, tf.reshape(kept, (1, 32)), (0, 0))
                return both, out


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class WriteKept:
    """A loop carries the tile on; a load lands in the carried copy, and the tile is read after."""

    @func
    def run(
        a: Tensor[(2, 32), "f32"],
    ) -> tuple[Tensor[(2, 32), "f32"], Tensor[(2, 32), "f32"]]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(2, 32), names=("warp", "lane")) as threads:
                with threads[0, :32] as _loader:
                    tile = tf.schedule((a[0, :], tf.zeros(TILE)), op=T.copy())
                out = tf.zeros(Tensor[(2, 32), "f32"])
                kept = tile
                for k in range(2):
                    with threads[1, :32] as _reader:
                        out = tf.insert_slice(out, tf.reshape(kept, (1, 32)), (k, 0))
                        kept = tf.reshape(tf.reshape(kept, (1, 32)), (32,))
                with threads[0, :32] as _loader:
                    fresh = tf.schedule((a[1, :], kept), op=T.copy())
                with threads[1, :32] as _reader:
                    both = tf.insert_slice(
                        tf.zeros(Tensor[(2, 32), "f32"]), tf.reshape(fresh, (1, 32)), (1, 0)
                    )
                    both = tf.insert_slice(both, tf.reshape(tile, (1, 32)), (0, 0))
                return both, out


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class CarryBeforeLoad:
    """A loop carries the tile on while its body loads into the tile itself each round."""

    @func
    def run(
        a: Tensor[(3, 32), "f32"],
    ) -> tuple[Tensor[(2, 32), "f32"], Tensor[(2, 32), "f32"]]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(2, 32), names=("warp", "lane")) as threads:
                with threads[0, :32] as _loader:
                    tile = tf.schedule((a[0, :], tf.zeros(TILE)), op=T.copy())
                out = tf.zeros(Tensor[(2, 32), "f32"])
                got = tf.zeros(Tensor[(2, 32), "f32"])
                kept = tile
                for k in range(2):
                    with threads[1, :32] as _reader:
                        out = tf.insert_slice(out, tf.reshape(kept, (1, 32)), (k, 0))
                        kept = tf.reshape(tf.reshape(kept, (1, 32)), (32,))
                    with threads[0, :32] as _loader:
                        fresh = tf.schedule((a[k + 1, :], tile), op=T.copy())
                    with threads[1, :32] as _reader:
                        got = tf.insert_slice(got, tf.reshape(fresh, (1, 32)), (k, 0))
                return out, got


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class CarryOtherHalf:
    """A loop carries a tile's upper half on, and reads a load the lower half took first."""

    @func
    def run(a: Tensor[(32,), "f32"]) -> tuple[Tensor[(2, 32), "f32"], Tensor[(2, 32), "f32"]]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(2, 32), names=("warp", "lane")) as threads:
                with threads[0, :32] as _loader:
                    tile = tf.zeros(WIDE)
                with threads[0, :32] as _loader:
                    lower = tf.schedule((a, tile[0:32]), op=T.copy())
                out = tf.zeros(Tensor[(2, 32), "f32"])
                got = tf.zeros(Tensor[(2, 32), "f32"])
                kept = tile[32:64]
                for k in range(2):
                    with threads[1, :32] as _reader:
                        out = tf.insert_slice(out, tf.reshape(kept, (1, 32)), (k, 0))
                        got = tf.insert_slice(got, tf.reshape(lower, (1, 32)), (k, 0))
                        kept = tf.reshape(tf.reshape(kept, (1, 32)), (32,))
                return out, got


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class CrossHalves:
    """Loads fill the halves of one tile; a reader takes the old lower half beside the new upper."""

    @func
    def run(
        a: Tensor[(2, 32), "f32"],
    ) -> tuple[Tensor[(1, 32), "f32"], Tensor[(2, 32), "f32"]]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(2, 32), names=("warp", "lane")) as threads:
                with threads[0, :32] as _loader:
                    tile = tf.zeros(WIDE)
                with threads[0, :32] as _loader:
                    lower = tf.schedule((a[0, :], tile[0:32]), op=T.copy())
                with threads[1, :32] as _reader:
                    first = tf.insert_slice(
                        tf.zeros(Tensor[(1, 32), "f32"]), tf.reshape(lower, (1, 32)), (0, 0)
                    )
                with threads[0, :32] as _loader:
                    upper = tf.schedule((a[1, :], tile[32:64]), op=T.copy())
                with threads[1, :32] as _reader:
                    mixed = tf.insert_slice(
                        tf.zeros(Tensor[(2, 32), "f32"]), tf.reshape(tile[0:32], (1, 32)), (0, 0)
                    )
                    mixed = tf.insert_slice(mixed, tf.reshape(upper, (1, 32)), (1, 0))
                return first, mixed


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class OneTrip:
    """A one round loop overwrites the carried tile; a later read goes through it."""

    @func
    def run(
        a: Tensor[(2, 32), "f32"],
    ) -> tuple[Tensor[(1, 32), "f32"], Tensor[(1, 32), "f32"]]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(2, 32), names=("warp", "lane")) as threads:
                with threads[0, :32] as _loader:
                    tile = tf.schedule((a[0, :], tf.zeros(TILE)), op=T.copy())
                kept = tile
                for _k in range(1):
                    with threads[0, :32] as _loader:
                        kept = tf.schedule((a[1, :], kept), op=T.copy())
                with threads[0, :32] as _loader:
                    fresh = tf.schedule((a[1, :], tile), op=T.copy())
                with threads[1, :32] as _reader:
                    new = tf.insert_slice(
                        tf.zeros(Tensor[(1, 32), "f32"]), tf.reshape(fresh, (1, 32)), (0, 0)
                    )
                with threads[1, :32] as _reader:
                    old = tf.insert_slice(
                        tf.zeros(Tensor[(1, 32), "f32"]), tf.reshape(kept, (1, 32)), (0, 0)
                    )
                return new, old


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class DisjointHandoff:
    """The body overwrites a tile's upper half and hands its lower half on as it was."""

    @func
    def run(
        a: Tensor[(2, 64), "f32"],
    ) -> tuple[Tensor[(2, 32), "f32"], Tensor[(2, 32), "f32"]]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(2, 32), names=("warp", "lane")) as threads:
                with threads[0, :32] as _loader:
                    tile = tf.schedule((a[0, :], tf.zeros(WIDE)), op=T.copy())
                got = tf.zeros(Tensor[(2, 32), "f32"])
                out = tf.zeros(Tensor[(2, 32), "f32"])
                kept = tile[0:32]
                for k in range(2):
                    with threads[1, :32] as _reader:
                        out = tf.insert_slice(out, tf.reshape(kept, (1, 32)), (k, 0))
                    with threads[0, :32] as _loader:
                        upper = tf.schedule((a[1, 32:64], tile[32:64]), op=T.copy())
                    with threads[1, :32] as _reader:
                        got = tf.insert_slice(got, tf.reshape(upper, (1, 32)), (k, 0))
                        kept = tf.reshape(tf.reshape(tile[0:32], (1, 32)), (32,))
                return got, out


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class YieldOld:
    """The body overwrites the carried tile in place, then hands the tile on as it was."""

    @func
    def run(
        a: Tensor[(2, 32), "f32"],
    ) -> tuple[Tensor[(2, 32), "f32"], Tensor[(2, 32), "f32"]]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(2, 32), names=("warp", "lane")) as threads:
                with threads[0, :32] as _loader:
                    tile = tf.schedule((a[0, :], tf.zeros(TILE)), op=T.copy())
                got = tf.zeros(Tensor[(2, 32), "f32"])
                out = tf.zeros(Tensor[(2, 32), "f32"])
                kept = tile
                for k in range(2):
                    with threads[1, :32] as _reader:
                        out = tf.insert_slice(out, tf.reshape(kept, (1, 32)), (k, 0))
                    with threads[0, :32] as _loader:
                        fresh = tf.schedule((a[1, :], kept), op=T.copy())
                    with threads[1, :32] as _reader:
                        got = tf.insert_slice(got, tf.reshape(fresh, (1, 32)), (k, 0))
                        kept = tf.reshape(tf.reshape(kept, (1, 32)), (32,))
                return got, out


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class PartialReload:
    """A region reads the old tile and reloads its lower half, after a full load took the tile."""

    @func
    def run(
        a: Tensor[(2, 64), "f32"],
    ) -> tuple[Tensor[(1, 32), "f32"], Tensor[(2, 32), "f32"]]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(2, 32), names=("warp", "lane")) as threads:
                with threads[0, :32] as _loader:
                    tile = tf.zeros(WIDE)
                with threads[0, :32] as _loader:
                    full = tf.schedule((a[0, :], tile), op=T.copy())
                with threads[1, :32] as _reader:
                    upper = tf.insert_slice(
                        tf.zeros(Tensor[(1, 32), "f32"]), tf.reshape(full[32:64], (1, 32)), (0, 0)
                    )
                with threads[0, :32] as _loader:
                    both = tf.insert_slice(
                        tf.zeros(Tensor[(2, 32), "f32"]), tf.reshape(tile[32:64], (1, 32)), (0, 0)
                    )
                    lower = tf.schedule((a[1, 0:32], tile[0:32]), op=T.copy())
                with threads[1, :32] as _reader:
                    both = tf.insert_slice(both, tf.reshape(lower, (1, 32)), (1, 0))
                return upper, both


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class ReturnOld:
    """The function returns a row it overwrites, beside a copy of what the overwrite stored."""

    @func
    def run(a: Tensor[(32,), "f32"]) -> tuple[Tensor[(32,), "f32"], Tensor[(32,), "f32"]]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(2, 32), names=("warp", "lane")) as threads:
                with threads[0, :32] as _loader:
                    tile = tf.schedule((a, tf.zeros(TILE)), op=T.copy())
                row = tf.zeros(Tensor[(32,), "f32"])
                with threads[0, :32] as _loader:
                    stored = tf.schedule((tile, row), op=T.copy())
                with threads[1, :32] as _reader:
                    copied = tf.insert_slice(tf.zeros(Tensor[(32,), "f32"]), stored, (0,))
                return row, copied
