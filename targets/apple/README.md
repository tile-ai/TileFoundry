# TileFoundry Apple targets

`tilefoundry-apple` provides the `amx` and `metal` targets for TileFoundry. It depends on
exactly the same version of `tilefoundry`.

```sh
pip install "tilefoundry[apple]"
```

For development and third-party target packaging, see
[Writing a target](https://github.com/tile-ai/TileFoundry/blob/main/docs/develop.md#6-writing-a-target).

Metal descriptions and logical CPU evaluators use
`MetalTarget("apple.m2_pro.gpu")`; the device is required explicitly.
The package describes matrix patterns, SIMD/quad collectives, atomics and
synchronization. The 16×32×16 source-derived contracts support static and logical
checks but are not M2 capabilities. No Metal codegen or GPU submission is provided.
See [Metal target](tilefoundry_apple/metal/docs/spec/target.md) and
[TIR](tilefoundry_apple/metal/docs/spec/tir.md).
