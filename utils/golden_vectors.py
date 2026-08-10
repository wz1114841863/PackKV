"""BRISK-KV Format v0 golden-vector generation and self-validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from utils.compute import (
    BucketScoreMethod,
    RepackMethod,
    apply_quant_metadata_token_permutation,
    bit_pack_decode_kv,
    bit_pack_encode,
    build_kv_bucket_permutation,
    decode_bucket_metadata,
    decode_compact_quant_metadata,
    dequantize_ints,
    encode_bucket_metadata,
    verify_compact_quant_metadata_roundtrip,
    verify_repack_bitstream_roundtrip,
)


FORMAT_NAME = "briskkv-format-v0"
DEFAULT_BLOCK_SIZE = 64
DEFAULT_PACK_SIZE = 16
DEFAULT_BUCKET_COUNT = 4
K_ZERO_POINT_BITS = 7
V_ZERO_POINT_BITS = 5
EXPONENT_BITS = 4


@dataclass(frozen=True)
class GoldenVectorCase:
    name: str
    description: str
    k_q: torch.Tensor
    v_q: torch.Tensor
    k_zero: torch.Tensor
    v_zero: torch.Tensor
    k_exponent: torch.Tensor
    v_exponent: torch.Tensor

    def validate(self) -> None:
        if self.k_q.shape != self.v_q.shape or self.k_q.ndim != 5:
            raise ValueError("K/V q tensors must have matching 5-D shapes")
        metadata = (self.k_zero, self.v_zero, self.k_exponent, self.v_exponent)
        if any(value.ndim != 5 for value in metadata):
            raise ValueError("zero/exponent tensors must be 5-D")
        if any(value.shape[2:4] != self.k_q.shape[2:4] for value in metadata):
            raise ValueError("metadata block/token dimensions must match q")
        if self.k_q.shape[3] != DEFAULT_BLOCK_SIZE:
            raise ValueError("Format v0 golden cases require 64-token blocks")


def _tensor_to_blocks(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(2, 3, 0, 1, 4).flatten(2, 4)


def _blocks_to_tensor(blocks: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    batch, heads, block_count, token_count, feature_dim = template.shape
    return blocks.reshape(
        block_count, token_count, batch, heads, feature_dim
    ).permute(2, 3, 0, 1, 4)


def _raw_tensor_bytes(tensor: torch.Tensor, dtype: str) -> bytes:
    array = tensor.detach().contiguous().cpu().numpy()
    return array.astype(np.dtype(dtype), copy=False).tobytes(order="C")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_entry(filename: str, data: bytes, role: str) -> dict:
    return {
        "file": filename,
        "role": role,
        "bytes": len(data),
        "sha256": _sha256(data),
    }


def _metadata_shape(value: torch.Tensor) -> list[int]:
    return [int(dim) for dim in value.shape]


def directed_cases() -> Tuple[GoldenVectorCase, ...]:
    token = torch.arange(DEFAULT_BLOCK_SIZE, dtype=torch.int64)

    nonidentity_k = ((63 - token) % 49).reshape(1, 1, 1, 64, 1)
    nonidentity_k = torch.cat(
        [nonidentity_k, (nonidentity_k * 3 + 5) % 49,
         (nonidentity_k * 5 + 7) % 49, (nonidentity_k * 7 + 11) % 49],
        dim=4,
    )
    nonidentity_v = torch.stack(
        [(token * factor + offset) % 16 for factor, offset in ((1, 1), (3, 2), (5, 4), (7, 8))],
        dim=1,
    ).reshape(1, 1, 1, 64, 4)
    k_zero = -((token % 29) + 6).reshape(1, 1, 1, 64, 1)
    v_zero = -((token % 12) + 1).reshape(1, 1, 1, 64, 1)
    k_exponent = ((token % 8) - 6).reshape(1, 1, 1, 64, 1)
    v_exponent = (((63 - token) % 8) - 3).reshape(1, 1, 1, 64, 1)

    constant_k = torch.full((1, 1, 1, 64, 4), 17, dtype=torch.int64)
    constant_v = torch.full((1, 1, 1, 64, 4), 6, dtype=torch.int64)
    constant_zero_k = torch.full((1, 1, 1, 64, 1), -12, dtype=torch.int64)
    constant_zero_v = torch.full((1, 1, 1, 64, 1), -4, dtype=torch.int64)
    constant_exponent_k = torch.full((1, 1, 1, 64, 1), -2, dtype=torch.int64)
    constant_exponent_v = torch.full((1, 1, 1, 64, 1), -1, dtype=torch.int64)

    generator = torch.Generator().manual_seed(20260809)
    random_k = torch.randint(0, 49, (1, 1, 2, 64, 4), generator=generator)
    random_v = torch.randint(0, 16, (1, 1, 2, 64, 4), generator=generator)
    random_k_zero = -torch.randint(
        6, 35, (1, 1, 2, 64, 1), generator=generator
    )
    random_v_zero = -torch.randint(
        1, 13, (1, 1, 2, 64, 1), generator=generator
    )
    random_k_exponent = torch.randint(
        -6, 5, (1, 1, 2, 64, 1), generator=generator
    )
    random_v_exponent = torch.randint(
        -6, 5, (1, 1, 2, 64, 1), generator=generator
    )

    return (
        GoldenVectorCase(
            name="directed_nonidentity",
            description="Non-identity k_sum bucket permutation with token-varying metadata.",
            k_q=nonidentity_k,
            v_q=nonidentity_v,
            k_zero=k_zero,
            v_zero=v_zero,
            k_exponent=k_exponent,
            v_exponent=v_exponent,
        ),
        GoldenVectorCase(
            name="directed_width0",
            description="All q values are constant; every pack has dynamic width zero.",
            k_q=constant_k,
            v_q=constant_v,
            k_zero=constant_zero_k,
            v_zero=constant_zero_v,
            k_exponent=constant_exponent_k,
            v_exponent=constant_exponent_v,
        ),
        GoldenVectorCase(
            name="random_seed_20260809",
            description="Two deterministic random blocks covering ordinary stream behavior.",
            k_q=random_k,
            v_q=random_v,
            k_zero=random_k_zero,
            v_zero=random_v_zero,
            k_exponent=random_k_exponent,
            v_exponent=random_v_exponent,
        ),
    )


def build_case_artifacts(case: GoldenVectorCase) -> Tuple[dict, Dict[str, bytes]]:
    """Build and self-check every byte stream for one Format v0 case."""
    case.validate()
    k_scale = torch.exp2(case.k_exponent.to(torch.float32))
    v_scale = torch.exp2(case.v_exponent.to(torch.float32))
    permutation, bucket_metadata = build_kv_bucket_permutation(
        case.k_q,
        case.v_q,
        DEFAULT_BUCKET_COUNT,
        BucketScoreMethod.K_SUM,
    )

    k_q_storage = apply_quant_metadata_token_permutation(case.k_q, permutation)
    v_q_storage = apply_quant_metadata_token_permutation(case.v_q, permutation)
    k_zero_storage = apply_quant_metadata_token_permutation(case.k_zero, permutation)
    v_zero_storage = apply_quant_metadata_token_permutation(case.v_zero, permutation)
    k_scale_storage = apply_quant_metadata_token_permutation(k_scale, permutation)
    v_scale_storage = apply_quant_metadata_token_permutation(v_scale, permutation)
    k_exponent_storage = apply_quant_metadata_token_permutation(
        case.k_exponent, permutation
    )
    v_exponent_storage = apply_quant_metadata_token_permutation(
        case.v_exponent, permutation
    )

    repacked_blocks = torch.cat(
        [_tensor_to_blocks(k_q_storage), _tensor_to_blocks(v_q_storage)], dim=2
    )
    k_stream, v_stream = bit_pack_encode(repacked_blocks, DEFAULT_PACK_SIZE)
    bucket_stream = encode_bucket_metadata(bucket_metadata)
    k_metadata_stream = verify_compact_quant_metadata_roundtrip(
        k_zero_storage,
        k_scale_storage,
        K_ZERO_POINT_BITS,
        EXPONENT_BITS,
    )
    v_metadata_stream = verify_compact_quant_metadata_roundtrip(
        v_zero_storage,
        v_scale_storage,
        V_ZERO_POINT_BITS,
        EXPONENT_BITS,
    )

    decoded_bucket = decode_bucket_metadata(
        bucket_stream,
        block_count=case.k_q.shape[2],
        tokens_per_block=case.k_q.shape[3],
        bucket_count=DEFAULT_BUCKET_COUNT,
        score_method=BucketScoreMethod.K_SUM,
    )
    if decoded_bucket != bucket_metadata:
        raise AssertionError("golden bucket metadata failed self-validation")
    decoded_blocks = bit_pack_decode_kv(
        k_stream,
        v_stream,
        block_count=case.k_q.shape[2],
        tokens_per_block=case.k_q.shape[3],
    )
    if not torch.equal(decoded_blocks, repacked_blocks.to(torch.int64)):
        raise AssertionError("golden q bitstream failed self-validation")
    decoded_k_zero, decoded_k_scale = decode_compact_quant_metadata(
        k_metadata_stream
    )
    decoded_v_zero, decoded_v_scale = decode_compact_quant_metadata(
        v_metadata_stream
    )
    decoded_k_q = _blocks_to_tensor(
        decoded_blocks[:, :, : repacked_blocks.shape[2] // 2], case.k_q
    )
    decoded_v_q = _blocks_to_tensor(
        decoded_blocks[:, :, repacked_blocks.shape[2] // 2 :], case.v_q
    )
    decoded_k = dequantize_ints(
        decoded_k_q, decoded_k_zero, decoded_k_scale, False
    )
    decoded_v = dequantize_ints(
        decoded_v_q, decoded_v_zero, decoded_v_scale, False
    )
    expected_k = dequantize_ints(
        k_q_storage, k_zero_storage, k_scale_storage, False
    )
    expected_v = dequantize_ints(
        v_q_storage, v_zero_storage, v_scale_storage, False
    )
    if not torch.equal(decoded_k.to(torch.float32), expected_k.to(torch.float32)):
        raise AssertionError("golden K joint q/metadata decode mismatch")
    if not torch.equal(decoded_v.to(torch.float32), expected_v.to(torch.float32)):
        raise AssertionError("golden V joint q/metadata decode mismatch")

    audit = verify_repack_bitstream_roundtrip(
        case.k_q,
        case.v_q,
        DEFAULT_PACK_SIZE,
        RepackMethod.BUCKET,
        max_blocks=case.k_q.shape[2],
        bucket_count=DEFAULT_BUCKET_COUNT,
        bucket_score_method=BucketScoreMethod.K_SUM,
        k_quant_zero=case.k_zero,
        k_quant_scale=k_scale,
        v_quant_zero=case.v_zero,
        v_quant_scale=v_scale,
        k_zero_point_bits=K_ZERO_POINT_BITS,
        v_zero_point_bits=V_ZERO_POINT_BITS,
        exponent_bits=EXPONENT_BITS,
        fixed_bucket_permutation=permutation,
        fixed_repack_metadata=bucket_metadata,
    )
    if not audit.joint_dequant_verified:
        raise AssertionError("joint q/metadata audit did not execute")

    files: Dict[str, bytes] = {
        "input_k_q.bin": _raw_tensor_bytes(case.k_q, "u1"),
        "input_v_q.bin": _raw_tensor_bytes(case.v_q, "u1"),
        "input_k_zero.bin": _raw_tensor_bytes(case.k_zero, "i1"),
        "input_v_zero.bin": _raw_tensor_bytes(case.v_zero, "i1"),
        "input_k_exponent.bin": _raw_tensor_bytes(case.k_exponent, "i1"),
        "input_v_exponent.bin": _raw_tensor_bytes(case.v_exponent, "i1"),
        "bucket_counts.bin": bucket_stream,
        "k_zero_points.bin": k_metadata_stream.zero_points,
        "k_exponents.bin": k_metadata_stream.exponents,
        "v_zero_points.bin": v_metadata_stream.zero_points,
        "v_exponents.bin": v_metadata_stream.exponents,
        "k_pack_mins.bin": k_stream.pack_mins,
        "k_encode_lengths.bin": k_stream.encode_lengths,
        "k_payload.bin": k_stream.payload,
        "v_pack_mins.bin": v_stream.pack_mins,
        "v_encode_lengths.bin": v_stream.encode_lengths,
        "v_payload.bin": v_stream.payload,
        "expected_permutation.bin": _raw_tensor_bytes(permutation, "u1"),
        "expected_bucket_counts.bin": _raw_tensor_bytes(
            torch.tensor(bucket_metadata.bucket_counts), "u1"
        ),
        "expected_k_q.bin": _raw_tensor_bytes(k_q_storage, "u1"),
        "expected_v_q.bin": _raw_tensor_bytes(v_q_storage, "u1"),
        "expected_k_zero.bin": _raw_tensor_bytes(k_zero_storage, "i1"),
        "expected_v_zero.bin": _raw_tensor_bytes(v_zero_storage, "i1"),
        "expected_k_exponent.bin": _raw_tensor_bytes(k_exponent_storage, "i1"),
        "expected_v_exponent.bin": _raw_tensor_bytes(v_exponent_storage, "i1"),
        "expected_k_dequant_f32.bin": _raw_tensor_bytes(expected_k, "<f4"),
        "expected_v_dequant_f32.bin": _raw_tensor_bytes(expected_v, "<f4"),
    }

    roles = {
        name: (
            "encoder_input" if name.startswith("input_") else
            "decoder_input" if name in {
                "bucket_counts.bin", "k_zero_points.bin", "k_exponents.bin",
                "v_zero_points.bin", "v_exponents.bin", "k_pack_mins.bin",
                "k_encode_lengths.bin", "k_payload.bin", "v_pack_mins.bin",
                "v_encode_lengths.bin", "v_payload.bin",
            } else "expected_output"
        )
        for name in files
    }
    descriptor = {
        "format": FORMAT_NAME,
        "case": case.name,
        "description": case.description,
        "source": "deterministic_synthetic",
        "parameters": {
            "block_size": DEFAULT_BLOCK_SIZE,
            "pack_size": DEFAULT_PACK_SIZE,
            "bucket_count": DEFAULT_BUCKET_COUNT,
            "bucket_score_method": BucketScoreMethod.K_SUM.value,
            "k_zero_point_bits": K_ZERO_POINT_BITS,
            "v_zero_point_bits": V_ZERO_POINT_BITS,
            "exponent_bits": EXPONENT_BITS,
            "high_precision_zero_point": False,
            "byte_order": "little",
            "field_bit_order": "lsb_first",
        },
        "tensors": {
            "q_shape": _metadata_shape(case.k_q),
            "k_metadata_shape": _metadata_shape(case.k_zero),
            "v_metadata_shape": _metadata_shape(case.v_zero),
            "raw_input_q_dtype": "uint8",
            "raw_input_zero_dtype": "int8",
            "raw_input_exponent_dtype": "int8",
            "expected_dequant_dtype": "float32_le",
        },
        "bucket": {
            "count_field_bits": bucket_metadata.bucket_count_field_bits,
            "counts": [list(row) for row in bucket_metadata.bucket_counts],
        },
        "bitpack": {
            "k": {
                "token_count": k_stream.token_count,
                "feature_dim": k_stream.feature_dim,
                "pack_len": k_stream.pack_len,
                "padded_token_count": k_stream.padded_token_count,
                "code_value_bits": k_stream.code_value_bits,
                "encode_length_field_bits": k_stream.encode_length_field_bits,
                "signed_values": k_stream.signed_values,
            },
            "v": {
                "token_count": v_stream.token_count,
                "feature_dim": v_stream.feature_dim,
                "pack_len": v_stream.pack_len,
                "padded_token_count": v_stream.padded_token_count,
                "code_value_bits": v_stream.code_value_bits,
                "encode_length_field_bits": v_stream.encode_length_field_bits,
                "signed_values": v_stream.signed_values,
            },
        },
        "files": {
            name: _file_entry(name, data, roles[name])
            for name, data in sorted(files.items())
        },
        "validation": {
            "bucket_metadata_roundtrip": True,
            "bitpack_roundtrip": True,
            "compact_metadata_roundtrip": True,
            "joint_q_metadata_dequant_roundtrip": True,
        },
    }
    return descriptor, files


def export_cases(
    output_root: Path,
    cases: Iterable[GoldenVectorCase],
    overwrite: bool = False,
) -> dict:
    """Export case directories and a suite manifest below ``output_root``."""
    output_root = Path(output_root)
    cases = tuple(cases)
    manifest_path = output_root / "manifest.json"
    if not overwrite:
        existing = [output_root / case.name for case in cases if (output_root / case.name).exists()]
        if manifest_path.exists():
            existing.append(manifest_path)
        if existing:
            raise FileExistsError(
                f"golden-vector output already exists: {existing[0]}; use --overwrite"
            )
    built = []
    for case in cases:
        descriptor, files = build_case_artifacts(case)
        case_dir = output_root / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        for filename, data in files.items():
            (case_dir / filename).write_bytes(data)
        descriptor_bytes = (
            json.dumps(descriptor, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        (case_dir / "descriptor.json").write_bytes(descriptor_bytes)
        built.append(
            {
                "case": case.name,
                "directory": case.name,
                "descriptor_sha256": _sha256(descriptor_bytes),
                "file_count": len(files) + 1,
            }
        )

    manifest = {
        "format": FORMAT_NAME,
        "case_count": len(built),
        "cases": built,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_bytes)
    return manifest
