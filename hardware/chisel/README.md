# BRISK-KV Chisel

This directory contains the synthesizable Chisel reference implementation of
the BRISK-KV streaming codec. It is intentionally isolated from the Python
algorithm implementation in the repository root.

The normative component-stream contract is documented in
[`../docs/briskkv_format_v0.md`](../docs/briskkv_format_v0.md).

## Requirements

- JDK 17 or newer
- SBT
- Verilator for ChiselSim tests

## Commands

Run from this directory:

```bash
sbt test
```

The test suite checks the frozen Format v0 parameters and the first decoder
stage. `FixedWidthFieldUnpacker` and `CompactMetadataDecoder` are compared with
Python-generated vectors under both continuous traffic and randomized
Decoupled backpressure. `BucketCountDecoder` additionally checks independently
aligned three-byte block headers, reconstructs the fourth occupancy, and
rejects invalid padding or occupancy sums. Subsequent modules must add the same
kind of golden-vector test before they are treated as implemented.

## Source layout

```text
src/main/scala/briskkv/   Chisel modules and shared parameters
src/test/scala/briskkv/   ScalaTest/ChiselSim verification
src/test/resources/       Versioned deterministic Python golden vectors
```

SBT is the only supported build entry point for the first implementation. This
avoids dependency and plugin drift between multiple build systems.
