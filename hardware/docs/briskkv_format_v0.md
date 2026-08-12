# BRISK-KV Format v0

Status: initial hardware contract  
Reference implementation: `utils/compute.py`  
Byte order: little-endian, LSB-first field packing

## 1. Scope

Format v0 defines the named component streams exchanged between the Python
golden model and the BRISK-KV hardware codec. It does **not** define a monolithic
file header or an external-memory container. Stream lengths, tensor shape, and
the descriptors listed below are supplied out of band by the testbench or DMA
controller.

The current algorithm configuration is:

| Parameter | Format v0 value |
|---|---:|
| Quantization mode | shared K/V token-wise dimension |
| Scale method | nearest power of two |
| K relative scale | 0.03 |
| V relative scale | 0.10 |
| Base block | 64 tokens |
| Recent high-precision window | 192 tokens |
| Pack | 16 tokens |
| Repacking | stable four-bucket FIFO |
| Bucket score | sum of K quantized features (`k_sum`) |

The recent 192-token window remains in the model's high-precision cache and is
outside the encoded streams in this document.

## 2. Frozen field limits

| Field | Encoding | Width | Qualified observed range |
|---|---|---:|---:|
| K quantized value | unsigned | 6 bits | 0 to 48 |
| V quantized value | unsigned | 4 bits | 0 to 15 |
| K zero point | two's complement | 7 bits | -34 to -6 |
| V zero point | two's complement | 5 bits | -12 to -1 |
| Scale exponent | two's complement | 4 bits | -6 to 4 |
| Pack width | unsigned | 3 bits | 0 to 6 |
| Bucket ID | unsigned | 2 bits | 0 to 3 |
| Bucket count | unsigned | 7 bits | 0 to 64 |

These limits were qualified on Qwen3-4B, Qwen3-8B, and Llama-3.1-8B using
three non-overlapping 8192-token traces per model. They are not mathematical
bounds for arbitrary tensors. An encoder must report overflow and must never
truncate or saturate an out-of-range value silently.

## 3. Scalar bit encoding

All component streams use the same scalar packing rule implemented by
`_pack_unsigned_fields()`:

1. fields are visited in the order defined for the component;
2. bit 0 of each field is appended first;
3. the first appended bit occupies bit 0 of byte 0;
4. fields may cross byte boundaries;
5. the final byte is padded in its most-significant unused bits with zero;
6. padding is applied only at the component boundary unless stated otherwise.

Signed fields use fixed-width two's-complement representation before applying
the rule above. Decoders must reject a component with an unexpected byte count
or non-zero final padding bits.

## 4. Quantization metadata streams

K and V each have two independent metadata components:

```text
k_zero_points
k_exponents
v_zero_points
v_exponents
```

Zero point and scale tensors have identical shapes. For token-wise PackKV
quantization, the zero point and exponent belonging to a token are moved by
the **same block-local bucket permutation** as that token's K/V q values.
The resulting storage-order tensors are then flattened in PyTorch row-major
order. For each storage-order index `i`:

```text
K scale[i] = 2 ^ k_exponents[i]
V scale[i] = 2 ^ v_exponents[i]

K reconstructed = (K q + K zero) * 2 ^ K exponent
V reconstructed = (V q + V zero) * 2 ^ V exponent
```

In other words, `(K_q, K_zero, K_exponent, V_q, V_zero, V_exponent)` is one
logical token record during bucket routing. Encoding q in bucket order while
leaving token-wise metadata in original time order is invalid. No additional
permutation stream is required because all six fields share the same order.

Field widths are fixed at 7/4 bits for K zero/exponent and 5/4 bits for V
zero/exponent. Each of the four components is independently byte-aligned:

```text
zero_bytes     = ceil(parameter_count * zero_bits / 8)
exponent_bytes = ceil(parameter_count * 4 / 8)
```

Continuous scales and floating-point minima are invalid in Format v0.
GREEDY and MEDIAN repacking are also outside Format v0 because their current
software paths do not expose a token permutation that can be applied to the
per-token quantization metadata.

### 4.1 Write-side fixed-point quantizer contract

The synthesizable reference `KvWriteQuantizer` accepts one complete token
vector in signed Q12 fixed-point form. Its input width is an elaboration
parameter; the twelve fractional bits describe the upstream compute interface
and are not a serialized field. For fixed-point input `x`, it implements the integer
zero-point branch of `utils/compute.py:quant_ints()`:

```text
range = maximum(x) - minimum(x)
raw_scale = range * relative_scale
exponent = round_to_nearest_even(log2(raw_scale))
zero_point = round_to_nearest_even(minimum(x) / 2^exponent)
q = round_to_nearest_even(x / 2^exponent) - zero_point
```

K fixes `relative_scale = 3/100`; V fixes `relative_scale = 1/10`. The hardware
does not synthesize a logarithm or divider: constant FP32-qualified Q12
threshold comparators select the exponent, and power-of-two division is a
right shift with tie-to-even rounding. Metadata is emitted before q, and both streams carry
the same token tag. If exponent, zero point, or q does not fit the frozen field
limits, the token is rejected and neither stream is emitted. In particular, a
constant or sufficiently narrow fixed-point token normally selects an exponent
below -6 and is rejected by Format v0 rather than silently clamped.

This contract only proves equivalence for values already represented in Q12.
Conversion from a model producer's
BF16/FP16 output into that input format is outside Format v0 and requires a
separate numerical-accuracy qualification before a complete write path can be
claimed.

## 5. Stable four-bucket repacking

K and V for one token are concatenated before repacking, so both caches always
use the same token permutation.

For each 64-token base block, the encoder computes:

```text
score[token] = sum(K_q[token, all_features])
score_min    = min(score)
score_max    = max(score)
span         = score_max - score_min + 1

threshold[j] = score_min + ceil(span * j / 4),  j in {1, 2, 3}
bucket_id    = count(score >= threshold[j])
```

`ceil(span * j / 4)` is implemented as `(span * j + 3) >> 2`. Tokens are
written into four FIFO queues and read back in ascending bucket-ID order.
Relative token order inside each bucket is preserved.

The synthesizable `KvTokenJoinBucketRouter` implements the same behavior with
one 64-token block buffer. K and V metadata are accepted as an atomic pair, and
each K/V q feature is accepted as another atomic pair. Both sides must carry
the same token tag, feature index, and final marker. After classification, the
router scans original token indices for bucket 0 through bucket 3; this is
equivalent to four stable FIFOs but does not require a comparison sorting
network or a stored 64-entry permutation. The four occupancies are emitted as
one `BucketCountRecord`; the later byte encoder stores only the first three.

Router output is token-major: one joined metadata record precedes all joined
K/V q features for that routed token. Dynamic bit-packing requires
pack-major/feature-major/token-major order, so a subsequent pack transpose and
bit-packer stage is still required. The router alone is not a complete Format
v0 encoder.

The decoder does not need the thresholds. For each block it receives the first
three bucket occupancies as three 7-bit unsigned fields. The fourth is derived:

```text
count[3] = 64 - count[0] - count[1] - count[2]
```

The 21 payload bits are independently padded for every base block, producing a
three-byte bucket header per block. The decoder must reject a header whose first
three counts sum to more than 64.

Bucket counts delimit the repacked FIFO regions; they do not reconstruct the
original token order. Decode-time attention is valid because K and V share the
same permutation. This invariance must not be assumed for a causally masked
prefill computation.

## 6. K/V bit-packed streams

After bucket repacking, base blocks are flattened in block-major, token-major
order. If the total token count is not divisible by 16, the last token vector is
repeated only at the end of the flattened stream. `padded_token_count` is an
out-of-band descriptor and repeated padding is discarded after decode.

K and V are then encoded independently. For every `(pack, feature)` pair:

```text
minimum = min(q[0..15])
maximum = max(q[0..15])
width   = ceil(log2(maximum - minimum + 1))
delta   = q - minimum
```

`width = 0` represents a constant pack. It consumes no payload bits and all 16
decoded values equal `minimum`.

Each cache produces three independently byte-aligned components:

### 6.1 `pack_mins`

Iteration order is pack-major then feature-major. Every minimum uses the
Format v0 profile width: K uses 6 bits and V uses 4 bits. The field width must
not shrink when the values in a particular layer or test vector happen to fit
in fewer bits. Values are unsigned when `signed_values=false`; otherwise they
use two's complement.

### 6.2 `encode_lengths`

Iteration order is pack-major then feature-major. Every width uses
`encode_length_field_bits = ceil(log2(code_value_bits + 1))`, with a minimum of
one bit. Format v0 hardware must support up to three bits and reject a decoded
width larger than `code_value_bits`.

### 6.3 `payload`

Iteration order is pack-major, feature-major, then token-major. Each `delta`
uses the corresponding dynamic `width`. A zero-width field must have value
zero and appends no bits.

### 6.4 Exact fixed-point dequantization

The Chisel reference decoder represents every reconstructed value as a signed
18-bit integer with six implied fractional bits:

```text
fixed_raw = (q + zero_point) << (exponent + 6)
real_value = fixed_raw / 64
```

The qualified Format v0 exponent range is `[-6, 4]`, so the shift amount is
always `[0, 10]`. No right shift, rounding, floating-point multiplier, or
general integer multiplier is required. Encoded exponents outside `[-6, 4]`
are rejected even though a four-bit two's-complement field could represent a
wider range. Eighteen signed bits cover the full K 6-bit q / 7-bit zero-point
field range at exponent 4; V requires fewer bits.

Quantization metadata is token-major, but q payload decoding is
pack-major/feature-major/token-major. The reference hardware therefore buffers
the zero point and exponent for one 16-token pack and reuses those 16 records
for every feature descriptor in that pack. Repeated q values used to pad a
partial final pack are consumed but are not emitted by the dequantizer.

The optimized reference uses two such metadata banks. While one bank supplies
the current q pack, the other accepts the next pack's compact metadata. Four
64-bit counters report active cycles, emitted values, cycles waiting for the
current metadata bank, and cycles blocked by downstream backpressure. For the
committed deterministic 64-token/four-feature test with no external stalls,
ping-pong buffering reduces active cycles from 320 to 272 and metadata-wait
cycles from 64 to 16. These are module-level simulation results, not synthesized
frequency, latency, or end-to-end model throughput.

## 7. Out-of-band descriptor

Until a DMA/container header is designed, the testbench supplies these values:

```text
metadata_shape
parameter_count
block_count
tokens_per_block = 64
token_count
feature_dim
pack_len = 16
padded_token_count
code_value_bits for K and V (fixed to 6 and 4 by the Format v0 profile)
encode_length_field_bits for K and V
signed_values for K and V
byte length of every named component stream
```

The first Chisel implementation must not infer component boundaries from data.

### 7.1 Command-driven controller contract

`DecompressionPipelineController` accepts one tagged command containing:

```text
tag
token_count
feature_dim
descriptor_count
payload_byte_count
```

For pack size 16 it checks, before launching the byte-stream decoders:

```text
pack_count       = ceil(token_count / 16)
descriptor_count = pack_count * feature_dim
block_count      = ceil(token_count / 64)
```

An invalid command produces an error result without accepting component bytes.
A valid command moves through `idle -> launch -> running -> response`. Only one
command is active at a time. The tagged response is held under backpressure and
contains the derived counts plus a snapshot of the dequantizer performance
counters. Live progress reports completed values, feature descriptors, packs,
and 64-token blocks. K/V field widths remain elaboration-time parameters; a
future dual-stream controller will coordinate two instances rather than change
those widths at runtime.

### 7.2 Unified K/V command and completion barrier

`DualKvDecompressionController` fixes the two child configurations to the
Format v0 K and V widths. Its shared command contains:

```text
tag
token_count
feature_dim
descriptor_count
k_payload_byte_count
v_payload_byte_count
```

After shared geometry validation, the top launches K, V, and bucket-count
decoding on the same command lifecycle. Their input and output channels remain
independent Decoupled streams: backpressure on K does not force V or bucket
records to stop. The outer result is nevertheless synchronized by a three-way
completion barrier and is not issued until both child command results and the
last bucket record have completed. It checks both child errors and tags, shared
token counts, and the emitted bucket-record count. The response contains
separate K/V performance-counter snapshots.

This shared lifecycle and block index establish K/V/bucket correspondence; the
top does not reconstruct an original token permutation. Correctness still
requires the encoder to have applied the same block-local permutation to K, V,
and their token-wise metadata.

### 7.3 Compute-facing feature packet contract

The dequantizer naturally emits one scalar at a time in
pack-major/feature-major/token-major order. `AttentionFeaturePacketizer`
collects one descriptor into the following compute record:

```text
values[16]          signed 18-bit, six implied fractional bits
valid_tokens        1..16
descriptor_index
pack_index
feature_index
block_index
pack_within_block   0..3
last
```

For K, a consumer can broadcast the query scalar for `feature_index` and update
16 token-local QK accumulators with `values[0..15]`. For V, a consumer can pair
the 16 attention weights for the pack with the V feature-lane packet. A partial
last pack reports its true `valid_tokens`; all remaining lanes are zero and
must not contribute to computation.

K and V packet channels retain independent backpressure. The compute wrapper
holds the decompression result until both final feature packets have been
accepted, so command completion cannot race ahead of unconsumed compute data.
`QkDotProductAccumulator` consumes one query record and one K feature packet
atomically. The query record carries `value`, `feature_index`, `pack_index`, and
`last`, allowing the accumulator to validate both streams before consuming
them. For every accepted feature it performs 16 signed products in
parallel and updates one accumulator per token lane. The Format v0 compute
profile uses signed Q6 query/K inputs and signed 44-bit Q12 accumulators. The
module emits one `QkLogitPacket` after the final feature of each pack:

```text
logits[16]          signed 44-bit, 12 implied fractional bits
valid_tokens        copied from the K pack
pack_index
block_index
pack_within_block
last
```

Invalid lanes are forced to zero. The module validates query/K feature indices,
descriptor/pack/block indices, pack-local `valid_tokens`, and that `last` only
occurs on a final feature. Query and K channels cannot be consumed separately.
It holds the completed logit packet under backpressure and counts active cycles,
accepted packets, valid-lane MAC operations, query/key wait cycles, and
downstream stalls.

The default accumulator is sized for `feature_dim <= 256`; the module rejects a
larger runtime dimension. Both the maximum feature dimension and accumulator
width are elaboration parameters, with an elaboration-time width check against
the worst-case signed sum.

`QueryReplayBuffer` supplies that query stream. It loads the signed Q6 query
vector once into a synchronous memory and replays features `0..feature_dim-1`
for every K pack. A two-entry response queue hides the one-cycle memory latency,
holds records under backpressure, and sustains one query feature per cycle after
priming. Every replayed record includes its pack/feature indices and the final
marker. Runtime validation rejects zero pack counts, zero feature dimensions,
and dimensions above the elaboration-time memory capacity.

`QkComputePipeline` composes the replay buffer with the 16-lane accumulator. Its
command supplies `feature_dim` and `pack_count`; its input query vector contains
only `feature_dim` values, while its K input contains `pack_count * feature_dim`
packets. Completion is issued only after the final replayed query/K feature has
produced and drained its logit packet. The `1/sqrt(feature_dim)` scale, softmax,
attention-weight storage, and AV MAC remain separate later stages.

### 7.4 Full decompression-to-QK top

`BriskKvDecompressQkTop` connects the complete Format v0 byte-stream decoder to
the query-replay QK pipeline. One accepted command starts K/V dynamic unpack,
compact metadata decode, power-of-two dequantization, K/V feature packetization,
bucket-count decode, query loading, and QK accumulation. The external streams
are:

```text
inputs:  query Q6 values, K/V minima, widths, payloads, zero points, exponents,
         bucket-count bytes
outputs: 16-lane Q12 QK logits, V feature packets, bucket records, tagged result
```

K feature packets are consumed internally by QK. V feature packets and bucket
records remain independently backpressured because the later softmax/AV stage
has not yet been implemented. QK logits and V packets retain the same repacked
token order, so a later softmax can feed its weights directly to the V path
without restoring the original permutation.

The top-level completion barrier requires all compressed input streams to be
decoded, all V/bucket outputs to be consumed, and the final QK logit packet to
be consumed. Invalid descriptor geometry is handled by the existing compute
controller without starting Query replay; a feature dimension exceeding Query
memory capacity is rejected locally without consuming compressed inputs.

### 7.5 Fixed-point attention scale and streaming softmax

`AttentionScaleUnit` consumes the signed Q12 `QkLogitPacket` stream. For a
runtime feature dimension `d`, it selects the following unsigned Q18 constant
from an elaboration-time ROM:

```text
scale_q18[d] = round((1 / sqrt(d)) * 2^18)
scaled_q12   = symmetric_round(logit_q12 * scale_q18[d] / 2^18)
```

The current qualified runtime range is `1 <= d <= 256`. This replaces a
runtime square-root/divider with a small constant ROM. Scaling cannot increase
the absolute logit magnitude because `d >= 1`, so the signed 44-bit Q12 width
is preserved.

The implementation reuses four signed `44 x 20` multipliers across the four
groups of each 16-token packet. The Q18 scale is zero-extended before signed
multiplication, which preserves the `d = 1` value `2^18`. Scaling therefore
takes four compute cycles per packet instead of instantiating 16 wide
multipliers. Including the output handshake, the scaler accepts one packet
every five cycles. Since QK accumulation consumes `featureDim` feature packets
before producing one logit packet, this does not reduce steady-state throughput
at the current 128-feature synthesis point. Configurations with
`featureDim < 5` can backpressure QK; `scaleLanes` remains elaboration-time
configurable for that area/throughput trade-off.

All 64-bit performance counters are controlled by the elaboration-time
`enableStats` parameter. When disabled, functional datapaths, handshakes, and
the progress-port schema are unchanged, while every statistics output is tied
to zero so dead-code elimination removes the counter registers and increment
logic. This option is for matched stats-on/stats-off PPA ablation and does not
change the BRISK-KV on-wire format.

`StreamingSoftmax` normalizes across the complete token sequence, not within
each 16-token pack. Its hardware reference state machine has three memory
passes:

1. accept Q12 packets into logit SRAM and find the global maximum;
2. replay logits, evaluate `exp(x - maximum)`, accumulate the denominator, and
   store exponent packets;
3. calculate one shared reciprocal and stream normalized Q0.15 packets.

The exponential input is rounded to 1/16 steps and clamped to `[-8, 0]`. LUT
entries use unsigned Q16, including `exp(0) = 65536`. The denominator has
enough width for 16,384 tokens. One Q32 reciprocal is calculated per sequence
with an exact 33-cycle radix-2 restoring divider; the 16 lanes then use
parallel multiply-and-shift normalization rather than 16 parallel dividers.
Both the 16-lane signed maximum and exponent sum use balanced four-level
reduction trees. Invalid lanes in the final pack have zero exponent and zero
weight. The module stores and checks pack/block/final markers and holds every
output packet under backpressure.

These choices define the current Chisel reference arithmetic, not a Format v0
on-wire field. The iterative reciprocal increases per-sequence latency while
removing the wide single-cycle division path; it does not change packet
formats or numerical results.

### 7.6 Buffered V replay and 16-lane AV accumulation

The decompressor produces V in pack-major/feature-major order, while AV needs
to reuse one Softmax weight pack for every V feature. `VPacketBuffer` stores the
validated Q6 V packet stream in synchronous memory at:

```text
v_address = pack_index * feature_dim + feature_index
```

After all V descriptors are loaded, the buffer accepts tagged `(pack_index,
feature_index)` read requests. It permits one outstanding synchronous read and
holds the response under backpressure. Descriptor, pack, feature, block,
valid-lane, and final markers are checked while loading.

Only the 16 signed Q6 values are stored in the V SRAM (288 bits per entry).
The validated descriptor, pack, feature, block, valid-lane, and final fields
are reconstructed from the read request and transaction geometry. Likewise,
the Softmax weight SRAM stores only 16 Q0.15 weights (256 bits per entry);
position and boundary fields are validated before the write and derived again
from the addressed pack during replay.

`SoftmaxVAccumulator` stores the Q0.15 Softmax packets once, then traverses V in
feature-major/pack-major compute order. For each pack it performs 16 parallel
products and one adder-tree reduction:

```text
lane_product_q21 = weight_q15 * value_q6
feature_q21      = sum_over_all_valid_tokens(lane_product_q21)
```

The exact signed accumulator is 50 bits by default and no Q21-to-Q6 rounding
is performed in this stage. Invalid lanes in a partial final pack do not
contribute, even if their stored data is non-zero. Completion occurs only after
the final feature result has been accepted by the downstream consumer.

`SoftmaxVComputePipeline` joins V storage and AV accumulation and exports load,
read, response, output, MAC, wait, and backpressure counters. Its default
16,384-token by 256-feature address capacity is a functional upper bound. A
full instantiated buffer at that bound is large; synthesis evaluation must
compare it with tiled on-chip storage or a second compressed-V read/decode pass
after Softmax. This storage-policy choice does not change Format v0 encoding.

### 7.7 Compute-side output conversion and unified Attention top

`AvOutputQuantizer` converts the exact signed Q21 AV result back to the signed
18-bit Q6 convention used by the current compute-side interface:

```text
magnitude_q6 = round(abs(feature_q21) / 2^15)
signed_q6    = restore_sign(magnitude_q6)
output_q6    = saturate(signed_q6, -131072, 131071)
```

Rounding is symmetric round-to-nearest with half values moving away from zero.
Positive and negative saturation events are counted separately. Feature index
and final markers are validated, and the converted value is held under
backpressure. Saturation is an expected numerical event and is reported in
statistics rather than as a protocol error.

`BriskKvAttentionTop` connects the complete decode-time reference path:

```text
Format v0 K/V bytes
  -> dynamic unpack + compact metadata + Q6 dequantization
  -> 16-lane QK accumulation (Q12)
  -> Q18 reciprocal-square-root scale
  -> sequence-wide stable Softmax (Q0.15)
  -> buffered 16-lane AV accumulation (Q21)
  -> symmetric round and signed saturation (Q6)
```

One accepted command supplies the shared token count, feature dimension,
descriptor geometry, payload lengths, and tag. The QK, Softmax, V buffer, AV,
and output stages start only when the decompression geometry and configured
capacity are valid. Invalid commands return a tagged error without consuming
query or compressed payload streams.

K logits and V feature packets are internal to the unified top. Bucket records
remain independently visible for format auditing/control. The final tagged
result is held until decompression is complete, the bucket output has drained,
and the final Q6 attention feature has been accepted. Progress exposes the
existing decompression/QK, scale/Softmax, V/AV, and output-conversion counters.

## 8. Named components in one layer

Format v0 exposes the following independent byte arrays:

```text
k_zero_points
k_exponents
v_zero_points
v_exponents
bucket_counts
k_pack_mins
k_encode_lengths
k_payload
v_pack_mins
v_encode_lengths
v_payload
```

No concatenation order is normative in v0. A later container version may assign
base addresses or define a serialized layer header without changing the scalar
encoding rules above.

## 9. Decoder validation requirements

A conforming decoder or testbench must check:

- descriptor and byte lengths agree;
- unused final padding bits are zero;
- all four bucket counts are in `[0, 64]` and sum to 64;
- pack width does not exceed the cache's `code_value_bits`;
- signed zero points and exponents are sign-extended correctly;
- decoded exponent represents the exact shift `2^k`;
- decoded padding tokens are removed;
- K and V retain the same bucket permutation.

The Python functions `encode_compact_quant_metadata()`,
`decode_compact_quant_metadata()`, `encode_bucket_metadata()`,
`decode_bucket_metadata()`, `bit_pack_encode()`, and `bit_pack_decode()` are the
golden behavioral reference for Format v0.

## 10. Verification status

Completed in the Python reference model:

- compact metadata byte encode/decode;
- bucket-count metadata encode/decode;
- K/V bit-pack encode/decode;
- synthetic joint bucket-repacked q/zero/exponent dequantization round-trip;
- byte accounting including component alignment;
- prior component-level sampled round-trip on three models and three 8192-token
  traces per model.
- standalone deterministic directed/random Chisel golden-vector export;
- external three-model joint q/metadata audit reported passing (the server
  logs are not versioned in this repository).
- Chisel LSB-first fixed-width field unpacker;
- Chisel signed compact-metadata decoder, including randomized input/output
  backpressure and non-zero-padding rejection tests.
- Chisel four-bucket count decoder, including multi-block backpressure,
  per-block padding, and count-sum validation tests.
- Chisel write-side K/V token join and stable `k_sum` four-bucket router,
  including independent backpressure, exact metadata association, bucket-count
  validation, and whole-block rejection on a K/V tag mismatch.
- Chisel pack-descriptor decoder and runtime-width payload unpacker;
- integrated Chisel dynamic bit unpacker verified value-for-value against all
  committed directed/random K and V vectors, including zero-width packs and
  independent stream backpressure;
- dynamic payload byte-count validation, including truncated and extra-byte
  rejection tests.
- multiplier-free fixed-point power-of-two dequantizer with exponent-range
  validation;
- 16-token metadata alignment/reuse buffer and partial-pack padding removal;
- ping-pong metadata prefetch with cycle/output/stall counters and a retained
  single-buffer baseline for ablation;
- end-to-end K/V compact-metadata decode, dynamic unpack, and dequantization
  checked against committed Python float32 golden vectors.
- tagged single-stream decompression controller with command geometry checks,
  held completion response, live value/descriptor/pack/block progress, and
  performance-counter snapshot.
- unified K/V/bucket decompression top with independent stream backpressure,
  shared command geometry, three-way completion barrier, and separate K/V
  performance statistics.
- 16-token compute-side K/V feature packetization with partial-pack lane masks,
  explicit pack/block/feature indexing, independent channel backpressure, and
  completion-after-drain semantics.
- Q18 reciprocal-square-root attention scaling with symmetric Q12 rounding;
- sequence-wide stable softmax with synchronous logit/exponent memories,
  clamped Q16 exponential LUT, shared Q32 reciprocal, Q0.15 weights, partial
  pack handling, independent output backpressure, and performance counters.
- synchronous tagged V packet buffering plus 16-lane Q0.15-by-Q6 AV
  accumulation, exact Q21 feature outputs, feature/pack replay, partial-pack
  masking, completion-after-drain semantics, and wait/stall/MAC counters.
- symmetric Q21-to-Q6 rounding and signed saturation with separate clipping
  counters;
- unified compressed-byte-to-Q6 Attention top with shared command validation,
  internal QK/Softmax/AV streams, bucket audit output, and a final tagged
  completion/error barrier.
- parameterized split-SystemVerilog generation for the unified top, including
  a 1024-token/128-feature baseline, bodyless architectural-SRAM stubs with
  pre/post-DC black-box audits, and a CACTI-oriented depth/width/port inventory.

Not yet completed:

- export of selected real-model Cache blocks as committed/archived vectors;
- serialized DMA or memory-container header;
- synthesis, timing, area, power, and throughput evaluation.
