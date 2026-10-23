# TileFoundry AMX Spec — TIR

Apple AMX contracts are in `tilefoundry_amx.ir.tir.instructions`:
`T.amx_load_x` and `T.amx_load_y` transfer 32 FP16 values from global memory
to registers; `T.amx_fma16` accumulates their outer product into a 32-by-32
FP32 tile; `T.amx_store` transfers that tile to global memory. The accumulator
uses `Layout((32, (16, 2)), (32, (1, 16)))` to describe physical Z row order.
These operations execute on the `amx` mesh and require `amx_outer_product`.
Native wrappers for the corresponding loads, FMA and physical Z-row stores,
plus AMX state enable/disable, live in `targets/amx/tilefoundry_amx/include/tilefoundry/runtime/amx/instructions.h`.
