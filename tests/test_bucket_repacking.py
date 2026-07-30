import unittest

import torch

from utils.compute import (
    BucketScoreMethod,
    QuantMethod,
    RepackMethod,
    ScaleMethod,
    bucket_repacking,
    repack_and_encode,
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
        )

        restored = PackKVCacheConfig.from_str(str(config))
        self.assertEqual(restored, config)
        self.assertEqual(restored.bucket_count, 8)
        self.assertEqual(
            restored.bucket_score_method,
            BucketScoreMethod.KV_2D,
        )


if __name__ == "__main__":
    unittest.main()
