import unittest

import torch

from utils.compute import (
    QuantMode,
    ScaleMethod,
    decode_compact_quant_metadata,
    encode_compact_quant_metadata,
    profile_hardware_quantization,
    profile_integer_histogram,
    quant_ints,
    verify_compact_quant_metadata_roundtrip,
)


class HardwareFormatProfileTest(unittest.TestCase):
    def test_integer_histogram_reports_exact_tails_and_width(self):
        profile = profile_integer_histogram({-5: 1, 0: 998, 7: 1})

        self.assertEqual(profile.count, 1000)
        self.assertEqual(profile.min_value, -5)
        self.assertEqual(profile.p0001, -5)
        self.assertEqual(profile.p001, -5)
        self.assertEqual(profile.p999, 0)
        self.assertEqual(profile.p9999, 7)
        self.assertEqual(profile.max_value, 7)
        self.assertEqual(profile.required_bits, 4)

    def test_profile_separates_non_po2_and_non_integer_metadata(self):
        quant_int = torch.tensor([0.0, 1.0, 3.0, -2.0])
        zero = torch.tensor([-5.0, 0.0, 1.5, 7.0])
        scale = torch.tensor([0.125, 2.0, 3.0, 4.0])

        profile = profile_hardware_quantization(quant_int, zero, scale)

        self.assertEqual(profile.quantized.min_value, -2)
        self.assertEqual(profile.quantized.max_value, 3)
        self.assertEqual(profile.quantized.required_bits, 3)
        self.assertEqual(profile.zero_point.histogram, {-5: 1, 0: 1, 7: 1})
        self.assertEqual(profile.non_integer_zero_point_count, 1)
        self.assertEqual(profile.exponent.histogram, {-3: 1, 1: 1, 2: 1})
        self.assertEqual(profile.non_po2_scale_count, 1)

    def test_po2_nearest_quantization_has_integer_zero_and_exact_exponents(self):
        tensor = torch.linspace(-2.0, 3.0, steps=64).reshape(1, 1, 64, 1)
        quant_int, quant_zero, quant_scale = quant_ints(
            tensor=tensor,
            block_size=64,
            quant_scale_rel=0.1,
            quant_mode=QuantMode.TokenQuant,
            high_precision_zero_point=False,
            scale_method=ScaleMethod.PO2_NEAREST,
            record_k_stats=False,
        )

        profile = profile_hardware_quantization(
            quant_int, quant_zero, quant_scale
        )

        self.assertEqual(profile.non_integer_zero_point_count, 0)
        self.assertEqual(profile.non_po2_scale_count, 0)
        self.assertEqual(profile.exponent.count, quant_scale.numel())
        self.assertGreater(profile.quantized.required_bits, 0)
        self.assertGreater(profile.zero_point.required_bits, 0)

    def test_compact_metadata_byte_roundtrip_and_accounting(self):
        zero = torch.tensor([[-34.0, -12.0, -1.0, 7.0]])
        scale = torch.tensor([[2.0 ** -6, 0.5, 1.0, 16.0]])
        stream = verify_compact_quant_metadata_roundtrip(
            zero, scale, zero_point_bits=7, exponent_bits=4
        )
        decoded_zero, decoded_scale = decode_compact_quant_metadata(stream)

        torch.testing.assert_close(decoded_zero, zero.to(torch.int64))
        torch.testing.assert_close(decoded_scale, scale.to(torch.float32))
        self.assertEqual(len(stream.zero_points), 4)
        self.assertEqual(len(stream.exponents), 2)
        self.assertEqual(stream.total_bytes, 6)
        self.assertEqual(stream.alignment_bits, 4)

    def test_compact_metadata_rejects_field_overflow_and_continuous_scale(self):
        with self.assertRaisesRegex(ValueError, "does not fit"):
            encode_compact_quant_metadata(
                torch.tensor([-34.0]), torch.tensor([1.0]), 6, 4
            )
        with self.assertRaisesRegex(ValueError, "power-of-two"):
            encode_compact_quant_metadata(
                torch.tensor([-1.0]), torch.tensor([0.3]), 7, 4
            )


if __name__ == "__main__":
    unittest.main()
