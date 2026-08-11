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

`DecompressionPipelineController` wraps one compile-time-configured K or V
pipeline with a tagged Decoupled command/result protocol. It validates
`descriptorCount = ceil(tokenCount / 16) * featureDim` before accepting any
component bytes, generates the one-cycle engine start, tracks completed values,
descriptors, packs, and 64-token blocks, and holds the result until the consumer
accepts it. The result includes the command tag and a snapshot of all four
dequantizer performance counters.

`DualKvDecompressionController` is the unified Format v0 decompression top. A
single command supplies shared token/feature/descriptor geometry and separate
K/V payload byte lengths. It launches fixed K(6-bit q, 7-bit zero) and V(4-bit
q, 5-bit zero) child controllers together with `BucketCountDecoder`. K, V, and
bucket outputs retain independent Decoupled backpressure. A three-way completion
barrier produces one held result only after both child results and the final
bucket record complete; it reports separate K/V statistics and the bucket
record count.

`AttentionFeaturePacketizer` converts the scalar
pack/feature/token dequantization order into one 16-lane packet per feature.
Each packet carries `validTokens`, descriptor/pack/feature/block indices, and a
final marker; invalid lanes in a partial last pack are zero. This makes a K
packet suitable for broadcasting one query feature across 16 token lanes and a
V packet suitable for combining 16 attention weights with one V feature.
`BriskKvComputeInterface` wraps the dual decompressor with independent K/V
packetizers and does not release the shared completion result until both final
packets have actually been accepted by the compute-side consumers.

`QkDotProductAccumulator` is the first compute datapath stage. It atomically
accepts one signed Q6 query feature and one 16-lane signed Q6 K feature packet,
performs 16 parallel multiply-accumulates, and emits one signed 44-bit Q12 logit
packet after the final feature. It preserves pack/block indices and partial-pack
lane validity, checks stream ordering, holds results under backpressure, and
exports cycle, MAC, source-wait, and sink-stall counters. Query replay/storage,
attention scaling, and softmax are implemented as following pipeline stages.

`QueryReplayBuffer` stores one query vector in synchronous memory and replays it
for every K pack in pack-major/feature-major order. Its two-entry response queue
hides SRAM read latency and sustains one feature per cycle after priming while
preserving Decoupled backpressure. `QkComputePipeline` connects this replay path
to `QkDotProductAccumulator`, so software loads the query only once per command.
The joined pipeline checks query/K pack, feature, and final markers and exports
separate replay and MAC performance counters.

`BriskKvDecompressQkTop` is the integrated byte-stream-to-logit top. It connects
the dual K/V decompressor and packetizers to `QkComputePipeline`, consumes K
internally, and exposes QK logits plus independently backpressured V feature and
bucket streams. Its tagged result is held until decompression completes and the
final QK, V, and bucket outputs have all drained. Directed golden-vector tests
cover the complete compressed-byte-to-Q12-logit path and invalid command
geometry.

`AttentionScaleUnit` uses a 256-entry Q18 reciprocal-square-root ROM to apply
the runtime `1/sqrt(featureDim)` scale to Q12 logits. It uses symmetric
round-to-nearest arithmetic, preserves packet metadata, rejects unsupported
geometry, and reports packet/cycle/stall counters.

`StreamingSoftmax` performs sequence-wide, numerically stable fixed-point
softmax rather than normalizing each 16-token pack independently. It buffers
scaled logits in synchronous memory, finds the global maximum, evaluates a
Q16 `exp(x-max)` LUT at 1/16 steps over `[-8, 0]`, accumulates one denominator,
and normalizes through one shared Q32 reciprocal followed by 16 parallel
multipliers. The output is an independently backpressured Q0.15 weight packet.
`QkScaleSoftmaxPipeline` composes the scaler and softmax stages.

`VPacketBuffer` validates and stores the incoming pack-major/feature-major Q6
V stream in synchronous memory, then serves tagged `(pack, feature)` reads.
`SoftmaxVAccumulator` stores each Q0.15 weight pack once and replays it for
every V feature. Each compute step uses 16 signed V multipliers and an adder
tree; pack partial sums are accumulated into an exact signed Q21 result.
`SoftmaxVComputePipeline` joins the buffer and accumulator and retains full
Decoupled backpressure, partial-pack masking, completion, error, MAC, cycle,
wait, and stall accounting.

The default 16K-token/256-feature V-buffer parameters are a functional address
limit, not an area claim. Synthesis studies must compare full-head SRAM against
tiled buffering or a second compressed-V decode pass after Softmax.

## Source layout

```text
src/main/scala/briskkv/   Chisel modules and shared parameters
src/test/scala/briskkv/   ScalaTest/ChiselSim verification
src/test/resources/       Versioned deterministic Python golden vectors
```

SBT is the only supported build entry point for the first implementation. This
avoids dependency and plugin drift between multiple build systems.
