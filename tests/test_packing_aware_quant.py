import unittest

import torch

from utils.compute import (
    BucketScoreMethod,
    QuantMode,
    RepackMethod,
    ScaleMethod,
    packing_aware_quantize_kv,
    quant_ints,
    repack_and_encode,
)


class PackingAwareQuantTest(unittest.TestCase):
    def _run(self, k, v, budget=0.1):
        return packing_aware_quantize_kv(
            k,
            v,
            block_size=64,
            pack_size=16,
            k_quant_scale_rel=0.1,
            v_quant_scale_rel=0.1,
            k_quant_mode=QuantMode.TokenQuant,
            v_quant_mode=QuantMode.TokenQuant,
            high_precision_zero_point=False,
            k_error_budget=budget,
            v_error_budget=budget,
            bucket_count=4,
            bucket_score_method=BucketScoreMethod.K_SUM,
        )

    def test_selection_obeys_error_and_payload_guards(self):
        torch.manual_seed(7)
        k = torch.randn(1, 2, 128, 8)
        v = torch.randn(1, 2, 128, 8) * 0.4
        result = self._run(k, v, budget=0.25)
        k_q, _, k_scale, v_q, _, v_scale = result[:6]
        k_stats, v_stats = result[-2:]

        self.assertTrue(torch.isfinite(k_q).all())
        self.assertTrue(torch.isfinite(v_q).all())
        torch.testing.assert_close(
            torch.log2(k_scale.float()), torch.round(torch.log2(k_scale.float()))
        )
        torch.testing.assert_close(
            torch.log2(v_scale.float()), torch.round(torch.log2(v_scale.float()))
        )
        for stats in (k_stats, v_stats):
            self.assertEqual(stats.total_blocks, 2)
            self.assertEqual(stats.total_packs, 8)
            self.assertEqual(stats.error_budget_violations, 0)
            self.assertEqual(
                stats.ceil_selected_packs + stats.budget_rejected_beneficial_packs,
                stats.payload_beneficial_packs,
            )
            self.assertEqual(
                stats.positive_delta_candidates
                + stats.nonpositive_delta_selected_packs,
                stats.payload_beneficial_packs,
            )
            self.assertLessEqual(
                stats.selected_sse,
                1.25 * stats.nearest_sse + 1e-5 * max(1.0, stats.nearest_sse),
            )
            self.assertLessEqual(stats.error_budget_utilization, 1.00001)
            self.assertLessEqual(
                stats.payload_benefit_ceiling_bits, stats.selected_payload_bits
            )
            self.assertLessEqual(stats.selected_payload_bits, stats.nearest_payload_bits)
            self.assertGreaterEqual(stats.payload_bits_saved, 0)

    def test_fixed_permutation_is_shared_by_k_and_v(self):
        torch.manual_seed(19)
        k = torch.randn(1, 1, 64, 4)
        v = torch.randn(1, 1, 64, 4)
        result = self._run(k, v)
        k_q, _, _, v_q, _, _, permutation, metadata = result[:8]
        trace = []
        encoded_stats = repack_and_encode(
            k_q,
            v_q,
            pack_size=16,
            repack_method=RepackMethod.BUCKET,
            before_and_after_repacking=trace,
            bucket_count=4,
            bucket_score_method=BucketScoreMethod.K_SUM,
            fixed_bucket_permutation=permutation,
            fixed_repack_metadata=metadata,
            return_stats=True,
        )
        before, after = trace[0]
        expected = torch.gather(
            before, 1, permutation.unsqueeze(2).expand_as(before)
        )
        torch.testing.assert_close(after, expected)
        self.assertEqual(encoded_stats[2].payload_bits, result[-2].selected_payload_bits)
        self.assertEqual(encoded_stats[3].payload_bits, result[-1].selected_payload_bits)

    def test_zero_and_constant_blocks_do_not_create_nan(self):
        k = torch.zeros(1, 1, 64, 4)
        v = torch.full((1, 1, 64, 4), 3.0)
        result = self._run(k, v, budget=0.0)
        for tensor in result[:6]:
            self.assertTrue(torch.isfinite(tensor).all())
        self.assertEqual(result[-2].error_budget_violations, 0)
        self.assertEqual(result[-1].error_budget_violations, 0)

    def test_standalone_entry_rejects_joint_method(self):
        tensor = torch.zeros(1, 1, 64, 4)
        with self.assertRaisesRegex(ValueError, "joint K/V"):
            quant_ints(
                tensor,
                64,
                0.1,
                QuantMode.TokenQuant,
                False,
                ScaleMethod.PO2_PACK_AWARE,
            )


if __name__ == "__main__":
    unittest.main()
