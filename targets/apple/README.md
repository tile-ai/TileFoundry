# TileFoundry Apple targets

`tilefoundry-apple` provides the `amx` and `metal` targets for TileFoundry. It depends on
exactly the same version of `tilefoundry`.

```sh
pip install "tilefoundry[apple]"
```

For development and third-party target packaging, see
[Writing a target](https://github.com/tile-ai/TileFoundry/blob/main/docs/develop.md#6-writing-a-target).

## Metal

The `metal` target describes an Apple GPU through its SIMD-group functions:
8 × 8 matrix load, store and products, and lane reductions, prefixes, shuffles
and ballots. It declares these as `T.metal.*` TIR instructions, matches them to
HIR, and lowers scheduled HIR to TIR.

```python
from tilefoundry_apple import metal

target = metal.Target("apple.m2_pro.gpu")
```

It has no code generation and no runtime: a `metal` program cannot run on a
GPU. The matrix lane map was measured on Apple GPU family 8; the Metal Shading
Language does not specify it. See `tilefoundry spec metal/target` and
`tilefoundry spec metal/tir`.
