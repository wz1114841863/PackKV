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

Not yet completed:

- Python export of standalone Chisel golden-vector files;
- rerun of the joint q/metadata audit on real model traces;
- Chisel metadata decoder;
- Chisel bucket decoder;
- Chisel bit unpacker and shift-based dequantizer;
- serialized DMA or memory-container header;
- synthesis, timing, area, power, and throughput evaluation.
