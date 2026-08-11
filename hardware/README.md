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
