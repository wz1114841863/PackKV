# BRISK-KV JIT-V area ablation

The existing `BriskKvSingleHeadTileTop` remains the **Full-V** reference. It
materializes the complete dequantized V token-by-feature array before AV
accumulation.

Two additional synthesis tops isolate the proposed changes:

| Variant | Top | V materialization | Decompressors |
| --- | --- | --- | --- |
| Full-V | `BriskKvSingleHeadTileTop` | full token-feature SRAM | independent K/V |
| JIT-V dual | `BriskKvJitVSingleHeadTileTop` | 2-packet queue, per-feature partial sums, Softmax-weight SRAM | independent K/V |
| JIT-V shared | `BriskKvSharedJitVSingleHeadTileTop` | same JIT-V buffers | one time-shared K-capable decoder plus lossless V metadata-width adapter |

JIT-V starts V replay as soon as the first Softmax weight packet is committed.
Remaining weight production overlaps V replay/decompression in native
pack-major order. A V packet can leave the two-entry queue only when its
matching weight packet is already resident, so the overlap does not introduce
a read-before-write hazard. JIT-V therefore does not require a full
random-access V array. The shared variant is an area-priority ablation: K and
V are not decoded concurrently, so the second decompressor is removed. V
q-minimum fields are widened from 4 to 6 bits and signed zero points from 5 to
7 bits before entering the shared decoder. Payload, encode-width and exponent
streams are unchanged.

At the default 1024-token, 128-feature configuration, the generated memory
inventories contain 4,012,288 external-SRAM bits for Full-V and 1,659,392 bits
for either JIT-V variant: a 2,352,896-bit (58.64%) reduction. A paired run with
the bundled CACTI executable at 22 nm, 2.0 ns and maximum 128-bit banking
reported:

| Variant | SRAM area (mm^2) | Leakage (mW) |
| --- | ---: | ---: |
| Full-V | 0.847445 | 50.692293 |
| JIT-V dual/shared | 0.343719 | 22.426537 |

This is a storage-only estimate. Shared-decoder logic savings and timing must
be obtained from DC; CACTI cannot distinguish the dual and shared variants.

## Cycle benchmark

Run the three architectures in separate simulator processes and merge their
cycle-level results with:

```bash
FEATURE_DIM=8 TOKEN_COUNTS=64,256,1024 \
  bash hardware/scripts/run_cycle_benchmark.sh
```

The default output is
`hardware/evaluation/results/cycle_benchmark/cycle_benchmark_summary.csv`.
The benchmark uses identical raw K/V and query values for all variants and
rejects a run unless their output checksums match. It reports write latency,
total attention latency, K completion, first and last Softmax weight packets,
first and last V values, first and last output, JIT-V input stalls, and
externally observed output stalls. `TOKEN_COUNTS` may be set to `64` for a
quick smoke test. Simulator runs must remain sequential because ChiselSim uses
one suite work directory.

For `feature_dim=8` without output backpressure, launching V at the first
weight packet instead of the last produced the following cycle results. All
three architectures had identical output checksums.

| Tokens | JIT-V dual before | JIT-V dual overlap | Saved | JIT-V shared before | JIT-V shared overlap | Saved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1682 | 1673 | 9 | 1687 | 1678 | 9 |
| 256 | 6368 | 6323 | 45 | 6373 | 6328 | 45 |
| 1024 | 25112 | 24923 | 189 | 25117 | 24928 | 189 |

The saved cycles exactly match `weights_loaded_cycle - first_weight_cycle`.
The optimization therefore hides the full remaining Softmax-weight production
tail. It does not remove the longer serialized V-decompression/AV-accumulation
portion of JIT-V latency.

## Functional validation

Run dual and shared end-to-end tests in separate sbt processes to avoid holding
two large simulator elaborations in one JVM:

```bash
cd hardware/chisel
sbt -Dbriskkv.sharedDecompressor=false \
  'testOnly briskkv.BriskKvJitVSingleHeadTileTopSpec'
sbt -Dbriskkv.sharedDecompressor=true \
  'testOnly briskkv.BriskKvJitVSingleHeadTileTopSpec'
sbt 'testOnly briskkv.JitVAccumulatorSpec briskkv.VMetadataWidthAdapterSpec'
```

The end-to-end test covers raw K/V input, quantization, bucket routing,
bit-packing, resident stream storage, decompression, fixed-point attention,
output backpressure and exact output comparison.

## RTL export

```bash
ENABLE_STATS=false bash hardware/scripts/generate_jit_v_tile_rtl.sh
ENABLE_STATS=false bash hardware/scripts/generate_shared_jit_v_tile_rtl.sh
```

Each output has `full/` and `dc_logic/` variants. Use `dc_logic/filelist.f` and
`dc_logic/memory_modules.tcl` for DC logic-only synthesis. Use
`full/memories.csv` for CACTI aggregation. Compare all three architectures at
identical `MAXIMUM_TOKENS`, `MAXIMUM_FEATURE_DIM`, PDK and clock constraint.
