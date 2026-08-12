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

`KvWriteQuantizer` is the buffered write-side inverse of this contract. It
accepts one Q12 fixed-point K or V token vector, finds the token-wise range, selects
the nearest power-of-two scale with constant comparison thresholds, and emits
integer-zero-point metadata followed by the q vector. K uses relative scale
`3/100`; V uses `1/10`. Signed division uses round-to-nearest-even so fixed-point
inputs match the PyTorch reference at exact ties. Tokens whose exponent, zero
point, or q values exceed Format v0 are rejected before either output stream is
made valid. The module is intentionally standalone: bucket routing and dynamic
bit-packing remain separate write-side stages.

`KvTokenJoinBucketRouter` is the next write-side stage. It atomically joins K
and V metadata and q features by token tag, buffers one 64-token block, and
computes `k_sum` from the K q vector. Three integer equal-width thresholds
classify tokens into four buckets. The output walks original token indices once
per bucket, implementing stable FIFO order without a comparison sorter or a
stored permutation. Routed K q, V q, K/V metadata, and the token tag remain one
record. A mismatched tag, feature order, marker, or metadata field rejects the
whole block before any bucket header or routed value becomes valid.

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
multipliers. Maximum and exponent-sum reductions are balanced four-level
trees; the exact reciprocal uses a 33-cycle radix-2 divider instead of a wide
single-cycle `/` path. The output is an independently backpressured Q0.15
weight packet.
`QkScaleSoftmaxPipeline` composes the scaler and softmax stages.

`VPacketBuffer` validates and stores the incoming pack-major/feature-major Q6
V stream in synchronous memory, then serves tagged `(pack, feature)` reads.
Its load-order checker advances explicit pack and feature counters instead of
dividing the descriptor index by the runtime feature dimension. Block and
within-block fields use shifts and bit slices because Format v0 fixes four
packs per 64-token block; no divider or modulo operator remains in this module.
`DecompressionPipelineController` similarly checks transaction completion from
the validated descriptor, pack, block, and remaining-token counters. It does
not rebuild `tokenCount * featureDim` on the result-error path.
`SoftmaxVAccumulator` stores each Q0.15 weight pack once and replays it for
every V feature. Each compute step uses 16 signed V multipliers and an adder
tree; pack partial sums are accumulated into an exact signed Q21 result.
The SRAM payloads contain values only: 288 bits for a 16-lane V packet and
256 bits for a 16-lane weight packet. Packet indexes, valid-lane counts, block
position, and final markers are validated at ingress and reconstructed from
the address and active geometry at readout.
`SoftmaxVComputePipeline` joins the buffer and accumulator and retains full
Decoupled backpressure, partial-pack masking, completion, error, MAC, cycle,
wait, and stall accounting.

The default 16K-token/256-feature V-buffer parameters are a functional address
limit, not an area claim. Synthesis studies must compare full-head SRAM against
tiled buffering or a second compressed-V decode pass after Softmax.

`AvOutputQuantizer` converts the exact signed Q21 AV feature to the signed Q6
compute-side format using symmetric round-to-nearest and signed saturation.
It validates feature ordering, holds output under backpressure, and counts
positive and negative saturation separately.

`BriskKvAttentionTop` is the unified compressed-byte-to-attention-output top.
One accepted command launches decompression/QK, scale/Softmax, V buffering/AV,
and output conversion. K logits and V features remain internal; bucket records
remain an independently backpressured audit/control output. The tagged command
result cannot complete until the final Q6 attention feature and bucket record
have drained. Commands beyond the shared feature/token capacity are rejected
without consuming query or compressed streams.

The unified-top regression runs the non-identity directed vector, the all-zero
dynamic-width vector, and the two-block deterministic random vector with
randomized backpressure. It checks the final Q6 result and audits exact
packet/MAC counts at every compute stage. The zero-width case has empty payload
streams while retaining the fixed Format v0 pack-minimum widths (K=6, V=4), so
it guards against both dual-stream stalls and stream/profile width mismatches.

## Source layout

```text
src/main/scala/briskkv/   Chisel modules and shared parameters
src/test/scala/briskkv/   ScalaTest/ChiselSim verification
src/test/resources/       Versioned deterministic Python golden vectors
```

SBT is the only supported build entry point for the first implementation. This
avoids dependency and plugin drift between multiple build systems.

## SystemVerilog generation

`GenerateBriskKvAttentionTop` emits split SystemVerilog for a parameterized
unified Attention top. The repository wrapper generates a functional variant
and a DC logic-only variant:

```bash
bash ../scripts/generate_attention_rtl.sh
```

The default point is `maximumTokens=1024`, `maximumFeatureDim=128`, with four
physical attention-scale lanes. Override these with `MAXIMUM_TOKENS`,
`MAXIMUM_FEATURE_DIM`, `SCALE_LANES`, and `ENABLE_STATS`. `ENABLE_STATS=false`
keeps the progress ports but ties them to zero, allowing synthesis to remove
the counter registers and increment logic. Generate matched stats-on/off RTL
for area ablation with:

```bash
bash ../scripts/generate_stats_ablation_rtl.sh
```

Each variant has its own manifest and output directory. In `dc_logic`, the five
architectural synchronous memories are replaced by port-compatible, bodyless
SystemVerilog stubs marked `syn_black_box`; no `reg Memory[...]` implementation
is retained. `memory_modules.tcl` provides the mandatory DC instance audit and
`memories.csv` records depth, width, ports, access mode, instance count, and
total bits for CACTI. The 2-entry Query response queue is deliberately retained
as synthesized control logic. Use `full`, not `dc_logic`, for RTL simulation.
