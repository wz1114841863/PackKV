import math
import unittest

import torch

from utils.compute import (
    QuantMode,
    ScaleMethod,
    dequantize_ints,
    quant_ints,
    quant_ints_2k,
)


class QuantizationCorrectnessTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260805)

    @staticmethod
    def _quantize(tensor, method, high_precision_zero_point=False):
        return quant_ints(
            tensor,
            block_size=64,
            quant_scale_rel=0.1,
            quant_mode=QuantMode.TokenQuant,
            high_precision_zero_point=high_precision_zero_point,
            scale_method=method,
            record_k_stats=False,
        )

    def test_po2_scale_rounding_relations(self):
        tensor = torch.linspace(-0.25, 0.75, 64).reshape(1, 1, 64, 1)
        _, _, continuous = self._quantize(tensor, ScaleMethod.CONTINUOUS)
        _, _, nearest = self._quantize(tensor, ScaleMethod.PO2_NEAREST)
        _, _, floor = self._quantize(tensor, ScaleMethod.PO2_FLOOR)
        _, _, ceil = self._quantize(tensor, ScaleMethod.PO2_CEIL)

        for scale in (nearest, floor, ceil):
            exponent = torch.log2(scale.float())
            torch.testing.assert_close(exponent, torch.round(exponent))
        self.assertTrue(torch.all(floor <= continuous))
        self.assertTrue(torch.all(continuous <= ceil))
        expected_nearest = torch.exp2(torch.round(torch.log2(continuous.float())))
        torch.testing.assert_close(nearest.float(), expected_nearest)

    def test_dequantization_matches_each_zero_point_semantics(self):
        tensor = torch.randn(1, 2, 128, 8)
        for high_precision in (False, True):
            for method in (
                ScaleMethod.CONTINUOUS,
                ScaleMethod.PO2_NEAREST,
                ScaleMethod.PO2_FLOOR,
                ScaleMethod.PO2_CEIL,
            ):
                with self.subTest(high_precision=high_precision, method=method.value):
                    quant, zero, scale = self._quantize(
                        tensor, method, high_precision
                    )
                    reconstructed = dequantize_ints(
                        quant, zero, scale, high_precision
                    )
                    expected = (
                        quant * scale + zero
                        if high_precision
                        else (quant + zero) * scale
                    )
                    torch.testing.assert_close(
                        reconstructed, expected, rtol=0, atol=0
                    )
                    self.assertTrue(torch.isfinite(reconstructed).all())

    def test_legacy_2k_entry_matches_nearest(self):
        tensor = torch.randn(1, 1, 64, 4)
        expected = self._quantize(tensor, ScaleMethod.PO2_NEAREST)
        actual = quant_ints_2k(
            tensor,
            block_size=64,
            quant_scale_rel=0.1,
            quant_mode=QuantMode.TokenQuant,
            high_precision_zero_point=False,
        )
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor)

    def test_zero_fp16_blocks_are_finite_for_all_standalone_methods(self):
        tensor = torch.zeros(1, 1, 64, 4, dtype=torch.float16)
        for high_precision in (False, True):
            for method in (
                ScaleMethod.CONTINUOUS,
                ScaleMethod.PO2_NEAREST,
                ScaleMethod.PO2_FLOOR,
                ScaleMethod.PO2_CEIL,
            ):
                with self.subTest(high_precision=high_precision, method=method.value):
                    quant, zero, scale = self._quantize(
                        tensor, method, high_precision
                    )
                    reconstructed = dequantize_ints(
                        quant, zero, scale, high_precision
                    )
                    self.assertTrue(torch.isfinite(quant).all())
                    self.assertTrue(torch.isfinite(zero).all())
                    self.assertTrue(torch.isfinite(scale).all())
                    torch.testing.assert_close(
                        reconstructed, tensor.reshape_as(reconstructed)
                    )

    def test_nonzero_constant_fp16_requires_representable_zero_point(self):
        tensor = torch.full((1, 1, 64, 4), 3.0, dtype=torch.float16)
        for method in (
            ScaleMethod.CONTINUOUS,
            ScaleMethod.PO2_NEAREST,
            ScaleMethod.PO2_FLOOR,
            ScaleMethod.PO2_CEIL,
        ):
            with self.subTest(method=method.value):
                with self.assertRaisesRegex(FloatingPointError, "metadata dtype"):
                    self._quantize(tensor, method, False)
                quant, zero, scale = self._quantize(tensor, method, True)
                reconstructed = dequantize_ints(quant, zero, scale, True)
                self.assertTrue(torch.isfinite(reconstructed).all())
                torch.testing.assert_close(
                    reconstructed, tensor.reshape_as(reconstructed)
                )

    def test_floor_can_require_one_more_payload_bit(self):
        vector = torch.linspace(0.0, 1.0, 64).reshape(1, 1, 1, 64)
        tensor = vector.repeat(1, 1, 64, 1)
        floor_quant, _, _ = self._quantize(tensor, ScaleMethod.PO2_FLOOR)
        ceil_quant, _, _ = self._quantize(tensor, ScaleMethod.PO2_CEIL)

        def payload_width(quant):
            levels = int((quant.max() - quant.min()).item()) + 1
            return math.ceil(math.log2(levels)) if levels > 1 else 0

        self.assertEqual(payload_width(floor_quant), 5)
        self.assertLessEqual(payload_width(ceil_quant), 4)

    def test_invalid_inputs_fail_explicitly(self):
        valid = torch.zeros(1, 1, 64, 4)
        with self.assertRaisesRegex(ValueError, "block_size must be positive"):
            quant_ints(valid, 0, 0.1, QuantMode.TokenQuant)
        with self.assertRaisesRegex(ValueError, "divisible"):
            quant_ints(valid[:, :, :63], 64, 0.1, QuantMode.TokenQuant)
        with self.assertRaisesRegex(ValueError, "finite positive"):
            quant_ints(valid, 64, 0.0, QuantMode.TokenQuant)
        with self.assertRaisesRegex(TypeError, "floating-point"):
            quant_ints(valid.to(torch.int32), 64, 0.1, QuantMode.TokenQuant)
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            invalid = valid.clone()
            invalid[0, 0, 0, 0] = float("nan")
            quant_ints(invalid, 64, 0.1, QuantMode.TokenQuant)
        with self.assertRaisesRegex(ValueError, "shape"):
            quant_ints(valid.squeeze(0), 64, 0.1, QuantMode.TokenQuant)


if __name__ == "__main__":
    unittest.main()
