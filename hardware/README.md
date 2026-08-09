# BRISK-KV Hardware

BRISK-KV is the hardware-oriented KV-cache codec developed alongside the
PackKV-based Python research code in this repository. The hardware directory is
kept separate so that algorithm experiments do not become coupled to RTL build
artifacts or simulator dependencies.

## Current status

- BRISK-KV Format v0 field limits are frozen as an initial hardware contract.
- The Python encoder/decoder is the golden reference.
- The Chisel project and Format v0 parameter tests are present.
- Synthesizable metadata, bucket, bit-unpack, and dequantization modules are not
  implemented yet.

## Layout

```text
hardware/
├── docs/       Hardware format and interface specifications
└── chisel/     SBT-based Chisel implementation and tests
```

## Verification rule

A hardware module is considered implemented only after it passes both:

1. directed boundary and backpressure tests in ScalaTest/ChiselSim;
2. byte-for-byte or value-for-value comparison with vectors produced by the
   Python golden model in `utils/compute.py`.

See [`docs/briskkv_format_v0.md`](docs/briskkv_format_v0.md) for the normative
Format v0 component-stream contract.
