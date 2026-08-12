# BRISK-KV Hardware

BRISK-KV is the hardware-oriented KV-cache codec developed alongside the
PackKV-based Python research code in this repository. The hardware directory is
kept separate so that algorithm experiments do not become coupled to RTL build
artifacts or simulator dependencies.

## Current status

- BRISK-KV Format v0 field limits are frozen as an initial hardware contract.
- The Python encoder/decoder is the golden reference.
- Deterministic directed/random Format v0 golden-vector export is available.
- The Chisel project and Format v0 parameter tests are present.
- The synthesizable LSB-first fixed-width unpacker and compact signed-metadata
  decoder are implemented and verified against Python golden vectors.
- The four-bucket count-header decoder is implemented with per-block padding
  and count-sum validation.
- The three-stream dynamic bit-unpack path is implemented and verified for K/V,
  zero-width packs, independent backpressure, and malformed payload lengths.
- Exact shift-based power-of-two dequantization is implemented with a signed
  18-bit, six-fractional-bit output, 16-token metadata reuse, and end-to-end
  Python golden-vector verification.
- A tagged command-driven pipeline controller validates stream geometry,
  launches one K or V decoder, tracks value/descriptor/pack/block progress,
  and returns a held completion record with performance counters.
- A shared K/V top-level launches the fixed Format v0 K and V controllers with
  one geometry/tag command, decodes bucket-count headers in parallel, permits
  independent K/V/bucket backpressure, and joins all three completion paths.
- A compute-facing adapter aggregates scalar feature-major K/V results into
  16-token lane packets with pack/block/feature indices, valid-lane counts,
  padding suppression, and completion delayed until both packet streams drain.
- A 16-lane QK datapath atomically joins one query feature with one K feature
  packet, accumulates exact Q12 logits across the feature dimension, preserves
  partial-pack validity, and reports source/sink stall and MAC counters.
- A synchronous Query replay buffer stores one Q6 query vector, replays it once
  per K pack at one feature per cycle after priming, and forms a complete
  query-replay plus QK compute pipeline without software-side query repetition.
- The integrated decompression-to-QK top accepts complete compressed K/V and
  metadata byte streams, consumes K through the QK datapath, exposes aligned V
  and bucket outputs, and delays its tagged result until every output drains.
- A hardware-oriented Q12-to-Q12 attention scaler uses a runtime feature-dimension
  reciprocal-square-root ROM, followed by a sequence-wide stable softmax with
  SRAM buffering, a clamped Q16 exponential LUT, one shared reciprocal, Q0.15
  output weights, Decoupled backpressure, and cycle/stall counters.
- A synchronous V packet store changes the decompressor's pack-major stream
  into tagged feature/pack replay. A 16-lane Q0.15-by-Q6 AV engine reuses each
  weight pack across V features and emits exact Q21 feature results with
  partial-pack masking and full progress/stall statistics.
- A symmetric round-and-saturate stage converts exact Q21 AV results back to
  signed Q6 while counting positive/negative clipping. The unified Attention
  top connects every compressed K/V byte stream through QK, stable Softmax,
  buffered AV, and Q6 output with one tagged completion/error barrier.

## Layout

```text
hardware/
├── docs/             Hardware format and interface specifications
├── golden_vectors/   Python-to-Chisel reference-vector transport
└── chisel/           SBT-based Chisel implementation and tests
```

## Verification rule

A hardware module is considered implemented only after it passes both:

1. directed boundary and backpressure tests in ScalaTest/ChiselSim;
2. byte-for-byte or value-for-value comparison with vectors produced by the
   Python golden model in `utils/compute.py`.

See [`docs/briskkv_format_v0.md`](docs/briskkv_format_v0.md) for the normative
Format v0 component-stream contract.

The write-side metadata path uses `FixedWidthFieldPacker` to serialize compact
fields LSB-first. `CompactKvMetadataEncoder` produces four independently
aligned K/V zero-point and exponent streams, while `BucketCountEncoder`
produces one independently aligned three-byte occupancy header for every
64-token block.

`BriskKvWriteEncoderTop` connects the complete write path from paired Q12 K/V
values through token-wise power-of-two quantization, stable four-bucket routing,
16-token transpose, dynamic bit-packing, and all eleven Format v0 component
byte streams. It sequences multiple full blocks and reports completion only
after every output stream has drained.

## RTL and synthesis baseline

Generate the initial 1024-token, 128-feature SystemVerilog point with:

```bash
bash hardware/scripts/generate_attention_rtl.sh
```

Generate the independent stats-off write-side quantization/repacking/encoding
point with:

```bash
MAXIMUM_TOKENS=1024 MAXIMUM_FEATURE_DIM=128 ENABLE_STATS=false \
  bash hardware/scripts/generate_write_encoder_rtl.sh
```

The default is the `v1` write quantizer used by the 2026081205 DC baseline.
Set `QUANT_ARCHITECTURE=v2` only when regenerating the three-stage
quantization-parameter ablation:

```bash
QUANT_ARCHITECTURE=v2 MAXIMUM_TOKENS=1024 \
  MAXIMUM_FEATURE_DIM=128 ENABLE_STATS=false \
  bash hardware/scripts/generate_write_encoder_rtl.sh
```

`MAXIMUM_TOKENS` must be a positive multiple of 64 and bounds the number of
full blocks in one write transaction. Format v0 count and index ports remain
32 bits wide, while the generated implementation sizes internal feature,
descriptor, and remaining-block counters from these configured maxima.

The two variants are deliberately retained and named independently. `v1`
selects the exponent and validates zero point/range in one parameter state.
`v2` registers exponent selection, zero-point calculation, and maximum-code
validation as three parameter states. `v2` adds two cycles per token while
preserving the Format v0 quantization and byte-stream contracts; it is an
ablation rather than the default design.

`v3` starts from the v1 single-stage schedule and replaces only the exponent
priority chain with a leading-one candidate plus one exact adjacent-threshold
correction. It therefore has zero extra parameter cycles relative to v1. Use
`QUANT_ARCHITECTURE=v3` to generate this optimization candidate; v1 remains
the default until matched DC results demonstrate a PPA benefit.

The write export contains `full` simulation RTL and `dc_logic` RTL with an
automatically discovered SRAM inventory and bodyless DC black-box stubs.

`full` retains behavioral memory modules for RTL lint/simulation.
`dc_logic` includes an auto-generated SRAM black-box list and a CACTI-oriented
memory inventory. Design Compiler must not be assumed to ignore inferred
storage automatically: the supplied DC script explicitly black-boxes the five
architectural SRAM modules before elaborating the top. Logic PPA and CACTI
memory PPA must be reported separately and then combined with a stated method.

Evaluate that exact inventory with the bundled CACTI wrapper:

```bash
bash hardware/scripts/evaluate_cacti.sh
```

The evaluator width-banks wide RTL words, preserves CACTI padding, and reports
area, leakage, per-access energy, latency, and initiation interval. See
[`evaluation/mem/README.md`](evaluation/mem/README.md) for assumptions and
parameter overrides. Dynamic workload energy still requires read/write access
counts; summing the per-access numbers alone is not an end-to-end energy result.

Before accepting a remote DC run as the logic baseline, validate its completion
marker, mandatory reports, and pre/post-compile SRAM instance counts with:

```bash
python3 hardware/evaluation/synthesis/dc_baseline_report.py \
  --report-dir <run>/outputs \
  --dc-log <run>/dc.log \
  --memory-modules hardware/rtl/generated/briskkv_attention_t1024_f128/dc_logic/memory_modules.tcl \
  --output <run>/dc_baseline.json
```

The checker exits non-zero for an invalid baseline and leaves report metrics as
`null` when the installed DC version uses an unrecognized label.
