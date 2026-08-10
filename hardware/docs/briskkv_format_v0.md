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
stream-local `code_value_bits` descriptor. Values are unsigned when
`signed_values=false`; otherwise they use two's complement.

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
code_value_bits for K and V
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
The current module defines the transport and indexing contract only: query
storage, QK MACs, softmax, attention-weight storage, and AV MACs are not yet
implemented.

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

Not yet completed:

- export of selected real-model Cache blocks as committed/archived vectors;
- serialized DMA or memory-container header;
- synthesis, timing, area, power, and throughput evaluation.
