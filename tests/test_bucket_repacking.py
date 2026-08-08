import unittest

import torch

from utils.compute import (
    BucketScoreMethod,
    QuantMethod,
    RepackMethod,
    ScaleMethod,
    bit_pack_decode_kv,
    bit_pack_encode,
    bit_pack_stats,
    bucket_repacking,
    decode_bucket_metadata,
    encode_bucket_metadata,
    quant_error_kv_bucket_repacked,
    repack_and_encode,
    verify_repack_bitstream_roundtrip,
)
from utils.config import PackKVCacheConfig


class BucketRepackingTest(unittest.TestCase):
    def test_default_score_method_preserves_combined_sum_baseline(self):
        torch.manual_seed(11)
        blocks = torch.randint(-4, 12, (2, 64, 8), dtype=torch.int32)

        default_output, default_metadata = bucket_repacking(
            blocks,
            num_buckets=4,
            return_metadata=True,
        )
        explicit_output, explicit_metadata = bucket_repacking(
            blocks,
            num_buckets=4,
            score_method=BucketScoreMethod.COMBINED_SUM,
            return_metadata=True,
        )

        torch.testing.assert_close(default_output, explicit_output)
        self.assertEqual(default_metadata, explicit_metadata)

    def test_power_of_two_boundaries_and_stable_fifo_order(self):
        # score 范围为 [0,7],4 桶边界应为 2/4/6.
        blocks = torch.tensor(
            [
                [
                    [2, 0],
                    [0, 0],
                    [3, 0],
                    [1, 0],
                    [6, 0],
                    [4, 0],
                    [7, 0],
                    [5, 0],
                ]
            ],
            dtype=torch.int32,
        )

        repacked, metadata = bucket_repacking(
            blocks, num_buckets=4, return_metadata=True
        )

        expected = torch.tensor(
            [
                [
                    [0, 0],
                    [1, 0],
                    [2, 0],
                    [3, 0],
                    [4, 0],
                    [5, 0],
                    [6, 0],
                    [7, 0],
                ]
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(repacked, expected)
        self.assertEqual(metadata.bucket_counts, ((2, 2, 2, 2),))

    def test_constant_scores_use_first_bucket_without_divide_by_zero(self):
        blocks = torch.full((2, 8, 4), 3, dtype=torch.int32)
        repacked, metadata = bucket_repacking(
            blocks, num_buckets=4, return_metadata=True
        )

        torch.testing.assert_close(repacked, blocks)
        self.assertEqual(metadata.bucket_counts, ((8, 0, 0, 0), (8, 0, 0, 0)))

    def test_kv_pairs_are_reordered_as_indivisible_rows(self):
        blocks = torch.tensor(
            [
                [
                    [10, 100, 1, 11],
                    [20, 200, 2, 22],
                    [30, 300, 3, 33],
                    [40, 400, 4, 44],
                ]
            ],
            dtype=torch.int32,
        )
        repacked = bucket_repacking(blocks, num_buckets=2)

        original_rows = sorted(tuple(row) for row in blocks[0].tolist())
        repacked_rows = sorted(tuple(row) for row in repacked[0].tolist())
        self.assertEqual(repacked_rows, original_rows)

    def test_kv_2d_builds_two_by_two_stable_buckets(self):
        blocks = torch.tensor(
            [
                [
                    [10, 0, 10, 0],
                    [0, 0, 10, 0],
                    [10, 0, 0, 0],
                    [0, 0, 0, 0],
                    [9, 0, 9, 0],
                    [1, 0, 9, 0],
                    [9, 0, 1, 0],
                    [1, 0, 1, 0],
                ]
            ],
            dtype=torch.int32,
        )
        repacked, metadata = bucket_repacking(
            blocks,
            num_buckets=4,
            score_method=BucketScoreMethod.KV_2D,
            return_metadata=True,
        )
        expected = torch.tensor(
            [
                [
                    [0, 0, 0, 0],
                    [1, 0, 1, 0],
                    [0, 0, 10, 0],
                    [1, 0, 9, 0],
                    [10, 0, 0, 0],
                    [9, 0, 1, 0],
                    [10, 0, 10, 0],
                    [9, 0, 9, 0],
                ]
            ],
            dtype=torch.int32,
        )

        torch.testing.assert_close(repacked, expected)
        self.assertEqual(metadata.bucket_counts, ((2, 2, 2, 2),))
        self.assertEqual(metadata.bucket_score_method, "kv_2d")
        self.assertEqual(metadata.k_subbucket_count, 2)
        self.assertEqual(metadata.v_subbucket_count, 2)

    def test_online_bucket_repacking_preserves_prefill_and_decode_attention(self):
        torch.manual_seed(23)
        k = torch.randn(1, 2, 64, 4)
        v = torch.randn(1, 2, 64, 4)
        (
            stored_k,
            stored_v,
            k_buffer,
            v_buffer,
            prefill_k,
            prefill_v,
        ) = quant_error_kv_bucket_repacked(
            None,
            None,
            None,
            None,
            k,
            v,
            block_size=64,
            recent_size=0,
            k_quant_scale_rel=0.03,
            v_quant_scale_rel=0.1,
            k_quant_mode=QuantMethod.PackKV.value[0],
            v_quant_mode=QuantMethod.PackKV.value[1],
            scale_method=ScaleMethod.PO2_NEAREST,
            bucket_count=4,
            bucket_score_method=BucketScoreMethod.K_SUM,
        )

        self.assertEqual(k_buffer.shape[2], 0)
        self.assertEqual(v_buffer.shape[2], 0)
        self.assertFalse(torch.equal(stored_k, prefill_k))
        self.assertFalse(torch.equal(stored_v, prefill_v))

        # Prefill 保持时间顺序；decode 单 query 对共同 K/V 置换不变。
        query = torch.randn(1, 2, 1, 4)

        def attention(query_, key_, value_):
            scores = torch.matmul(query_, key_.transpose(-1, -2))
            probs = torch.softmax(scores, dim=-1)
            return torch.matmul(probs, value_)

        torch.testing.assert_close(
            attention(query, stored_k, stored_v),
            attention(query, prefill_k, prefill_v),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_online_bucket_repacking_rejects_chunked_prefill(self):
        k = torch.randn(1, 1, 64, 2)
        v = torch.randn(1, 1, 64, 2)
        state = quant_error_kv_bucket_repacked(
            None,
            None,
            None,
            None,
            k,
            v,
            block_size=64,
            recent_size=0,
            k_quant_scale_rel=0.03,
            v_quant_scale_rel=0.1,
            k_quant_mode=QuantMethod.PackKV.value[0],
            v_quant_mode=QuantMethod.PackKV.value[1],
            scale_method=ScaleMethod.PO2_NEAREST,
        )
        with self.assertRaisesRegex(ValueError, "chunked prefill"):
            quant_error_kv_bucket_repacked(
                state[0],
                state[1],
                state[2],
                state[3],
                torch.randn(1, 1, 2, 2),
                torch.randn(1, 1, 2, 2),
                block_size=64,
                recent_size=0,
                k_quant_scale_rel=0.03,
                v_quant_scale_rel=0.1,
                k_quant_mode=QuantMethod.PackKV.value[0],
                v_quant_mode=QuantMethod.PackKV.value[1],
                scale_method=ScaleMethod.PO2_NEAREST,
            )

    def test_bucket_metadata_uses_three_counts_for_four_buckets(self):
        k_tensor = torch.arange(8, dtype=torch.int32).reshape(1, 1, 1, 8, 1)
        v_tensor = torch.flip(k_tensor, dims=(3,))
        result = repack_and_encode(
            k_tensor,
            v_tensor,
            pack_size=4,
            repack_method=RepackMethod.BUCKET,
            return_stats=True,
            bucket_count=4,
            return_repack_metadata=True,
        )
        metadata = result[-1]

        # 每桶计数覆盖 [0,8],需要4 bit;最后一桶计数可推导.
        self.assertEqual(metadata.bucket_count_field_bits, 4)
        self.assertEqual(metadata.bucket_metadata_bits, 3 * 4)
        self.assertEqual(metadata.bucket_metadata_bytes, 2)

    def test_bucket_metadata_aligns_each_block_independently(self):
        blocks = torch.arange(2 * 64 * 2, dtype=torch.int32).reshape(2, 64, 2)
        _, metadata = bucket_repacking(
            blocks,
            num_buckets=4,
            return_metadata=True,
        )

        # 每个 block:3 个计数 * 7 bit = 21 bit,独立对齐为3 bytes.
        self.assertEqual(metadata.bucket_metadata_bits, 2 * 21)
        self.assertEqual(metadata.bucket_metadata_bytes, 2 * 3)

    def test_bucket_metadata_byte_roundtrip_with_empty_buckets(self):
        blocks = torch.tensor(
            [
                [[0, 3], [0, 2], [0, 1], [0, 0], [0, -1], [0, -2], [0, -3], [0, -4]],
                [[5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7]],
            ],
            dtype=torch.int32,
        )
        _, metadata = bucket_repacking(
            blocks,
            num_buckets=4,
            score_method=BucketScoreMethod.K_SUM,
            return_metadata=True,
        )

        encoded = encode_bucket_metadata(metadata)
        decoded = decode_bucket_metadata(
            encoded,
            block_count=blocks.shape[0],
            tokens_per_block=blocks.shape[1],
            bucket_count=4,
            score_method=BucketScoreMethod.K_SUM,
        )

        self.assertEqual(len(encoded), metadata.bucket_metadata_bytes)
        self.assertEqual(decoded, metadata)
        self.assertEqual(encode_bucket_metadata(decoded), encoded)

    def test_real_bit_pack_roundtrip_matches_accounting(self):
        # 包含负数、零位宽常量列以及流末 padding。
        torch.manual_seed(31)
        blocks = torch.randint(-7, 13, (3, 5, 8), dtype=torch.int32)
        blocks[:, :, 1] = -3
        blocks[:, :, 6] = 4
        k_stats, v_stats = bit_pack_stats(blocks, pack_len=4)
        k_stream, v_stream = bit_pack_encode(blocks, pack_len=4)
        decoded = bit_pack_decode_kv(
            k_stream,
            v_stream,
            block_count=blocks.shape[0],
            tokens_per_block=blocks.shape[1],
        )

        torch.testing.assert_close(decoded, blocks.to(torch.int64))
        self.assertEqual(k_stream.total_bytes, k_stats.total_bytes)
        self.assertEqual(v_stream.total_bytes, v_stats.total_bytes)
        self.assertEqual(len(k_stream.payload), k_stats.payload_bytes)
        self.assertEqual(len(v_stream.payload), v_stats.payload_bytes)
        self.assertEqual(len(k_stream.pack_mins), k_stats.pack_min_bytes)
        self.assertEqual(len(v_stream.pack_mins), v_stats.pack_min_bytes)
        self.assertEqual(
            len(k_stream.encode_lengths), k_stats.encode_length_bytes
        )
        self.assertEqual(
            len(v_stream.encode_lengths), v_stats.encode_length_bytes
        )

    def test_bucket_repack_and_bit_pack_end_to_end_roundtrip(self):
        torch.manual_seed(37)
        original = torch.randint(-16, 24, (2, 64, 12), dtype=torch.int32)
        repacked, metadata = bucket_repacking(
            original,
            num_buckets=4,
            score_method=BucketScoreMethod.K_SUM,
            return_metadata=True,
        )
        metadata_bytes = encode_bucket_metadata(metadata)
        decoded_metadata = decode_bucket_metadata(
            metadata_bytes,
            block_count=2,
            tokens_per_block=64,
            bucket_count=4,
            score_method=BucketScoreMethod.K_SUM,
        )
        k_stream, v_stream = bit_pack_encode(repacked, pack_len=16)
        decoded_repacked = bit_pack_decode_kv(
            k_stream, v_stream, block_count=2, tokens_per_block=64
        )

        torch.testing.assert_close(decoded_repacked, repacked.to(torch.int64))
        self.assertEqual(decoded_metadata.bucket_counts, metadata.bucket_counts)
        # 每行仍是不可分割的 K/V token 对，编码/解码不改变共同置换。
        for block_idx in range(original.shape[0]):
            original_rows = sorted(tuple(row) for row in original[block_idx].tolist())
            decoded_rows = sorted(
                tuple(row) for row in decoded_repacked[block_idx].tolist()
            )
            self.assertEqual(decoded_rows, original_rows)

    def test_real_quant_tensor_roundtrip_audit_for_none_and_bucket(self):
        torch.manual_seed(41)
        # [B,H,block,token,D]，与 quant_ints 的真实返回形状一致。
        k_tensor = torch.randint(-5, 19, (1, 2, 3, 64, 4), dtype=torch.int32)
        v_tensor = torch.randint(-9, 12, (1, 2, 3, 64, 4), dtype=torch.int32)

        none_stats = verify_repack_bitstream_roundtrip(
            k_tensor,
            v_tensor,
            pack_size=16,
            repack_method=RepackMethod.NONE,
            max_blocks=2,
        )
        bucket_stats = verify_repack_bitstream_roundtrip(
            k_tensor,
            v_tensor,
            pack_size=16,
            repack_method=RepackMethod.BUCKET,
            max_blocks=2,
            bucket_count=4,
            bucket_score_method=BucketScoreMethod.K_SUM,
        )

        self.assertEqual(none_stats.verified_blocks, 2)
        self.assertEqual(none_stats.bucket_metadata_bytes, 0)
        self.assertEqual(bucket_stats.verified_blocks, 2)
        # block_size=64、4桶：每 block 3*7=21 bit，独立对齐为3 bytes。
        self.assertEqual(bucket_stats.bucket_metadata_bytes, 6)
        self.assertGreater(none_stats.total_bytes, 0)
        self.assertGreater(bucket_stats.total_bytes, 0)

    def test_rejects_non_power_of_two_bucket_count(self):
        blocks = torch.zeros((1, 8, 2), dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "power of two"):
            bucket_repacking(blocks, num_buckets=3)

    def test_kv_2d_rejects_fewer_than_four_buckets(self):
        blocks = torch.zeros((1, 8, 2), dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "at least 4"):
            bucket_repacking(
                blocks,
                num_buckets=2,
                score_method=BucketScoreMethod.KV_2D,
            )

    def test_config_roundtrip_preserves_bucket_count(self):
        config = PackKVCacheConfig(
            model_name="synthetic",
            quant_method=QuantMethod.PackKV,
            repack_method=RepackMethod.BUCKET,
            high_precision_zero_point=False,
            block_size=64,
            buffer_size=192,
            pack_size=16,
            k_quant_scale_rel=0.1,
            v_quant_scale_rel=0.1,
            scale_method=ScaleMethod.PO2_NEAREST,
            bucket_count=8,
            bucket_score_method=BucketScoreMethod.KV_2D,
            k_error_budget=0.2,
            v_error_budget=0.3,
        )

        restored = PackKVCacheConfig.from_str(str(config))
        self.assertEqual(restored, config)
        self.assertEqual(restored.bucket_count, 8)
        self.assertEqual(restored.k_error_budget, 0.2)
        self.assertEqual(restored.v_error_budget, 0.3)
        self.assertEqual(
            restored.bucket_score_method,
            BucketScoreMethod.KV_2D,
        )


if __name__ == "__main__":
    unittest.main()
