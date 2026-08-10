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

The test suite checks the frozen Format v0 parameters and the implemented
decoder stages. `FixedWidthFieldUnpacker` and `CompactMetadataDecoder` are compared with
Python-generated vectors under both continuous traffic and randomized
Decoupled backpressure. `BucketCountDecoder` additionally checks independently
aligned three-byte block headers, reconstructs the fourth occupancy, and
rejects invalid padding or occupancy sums. `PackDescriptorDecoder`,
`DynamicPayloadUnpacker`, and `DynamicBitUnpacker` reconstruct the K/V integer
streams from independent minimum, width, and payload byte streams. Their tests
cover directed and random vectors, independently stalled streams, constant
packs (`width=0`), truncated payloads, and extra payload bytes. Subsequent
modules must add the same kind of golden-vector test before they are treated as
implemented.

`Po2FixedPointDequantizer` implements `(q + zero) * 2^exponent` without a
multiplier. Format v0 output is an exact signed 18-bit integer with six implied
fractional bits (`real = fixedRaw / 64`). `PackMetadataDequantizer` buffers the
16 token-wise zero/exponent records needed to bridge metadata token order and
q's feature-major pack order. `KvStreamDequantizer` integrates compact metadata
decode, dynamic q unpack, metadata reuse, padding removal, and fixed-point
dequantization. Its tests compare the final output directly with the Python
float32 golden vectors.

`BufferedPackMetadataDequantizer` is the default metadata joiner in
`KvStreamDequantizer`. It uses two 16-entry zero/exponent banks so one pack can
be consumed while the next is prefetched. Both the single- and double-buffered
implementations expose active-cycle, output-value, metadata-stall, and
downstream-stall counters. The deterministic 64-token/four-feature comparison
uses 320 cycles and 64 metadata-stall cycles for the single buffer versus 272
cycles and 16 metadata-stall cycles for the ping-pong buffer.

## Source layout

```text
src/main/scala/briskkv/   Chisel modules and shared parameters
src/test/scala/briskkv/   ScalaTest/ChiselSim verification
src/test/resources/       Versioned deterministic Python golden vectors
```

SBT is the only supported build entry point for the first implementation. This
avoids dependency and plugin drift between multiple build systems.
